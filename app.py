from flask import Flask, request, jsonify
from flask_cors import CORS
from extensions import mongo, init_mongo
from models.lipSyncTask import LipSyncTask
import requests
import subprocess
import os
import logging
from datetime import datetime
import threading
import sys
import boto3
from dotenv import load_dotenv
load_dotenv()
print("🔍 [DEBUG] app.py starting...")

app = Flask(__name__)
CORS(app)
init_mongo(app)

print("🔍 [DEBUG] Flask app initialized")
print(f"🔍 [DEBUG] MongoDB initialized: {mongo}")

@app.teardown_appcontext
def teardown_mongo(exception):
    """Close MongoDB connection on app shutdown to prevent thread errors"""
    try:
        mongo.cx.close()
        print("🔍 [DEBUG] MongoDB connection closed")
    except Exception as e:
        print(f"🔍 [DEBUG] Error closing MongoDB connection: {e}")

logger = logging.getLogger(__name__)

# Upscaling service configuration
UPSCALING_SERVICE_PORT = 5002
AUDIO_SYNC_OFFSETS = [-0.12, -0.08, -0.04, 0, 0.04, 0.08, 0.12]

print(f"🔍 [DEBUG] Upscaling service port: {UPSCALING_SERVICE_PORT}")

@app.route('/api/upscale/task/<task_id>', methods=['POST'])
def upscale_task(task_id):
    """
    Start upscaling for a specific lip sync task
    """
    print(f"🔍 [DEBUG] upscale_task endpoint called with task_id: {task_id}")
    try:
        # Get task details from lip_sync_tasks collection
        print(f"🔍 [DEBUG] Fetching task from MongoDB with task_id: {task_id}")
        task_data = mongo.db.lip_sync_tasks.find_one({"task_id": task_id})
        print(f"🔍 [DEBUG] Task data found: {task_data is not None}")
        
        if not task_data:
            print(f"🔍 [DEBUG] Task {task_id} not found in database")
            return jsonify({
                "success": False,
                "error": f"Task {task_id} not found"
            }), 404
        
        task = LipSyncTask.from_dict(task_data)
        print(f"🔍 [DEBUG] Task object created. Status: {task.status}")
        
        print(f"🔍 [DEBUG] Task status is completed. Output URLs count: {len(task.output_s3_urls)}")
        
        if not task.output_s3_urls:
            print(f"🔍 [DEBUG] No output S3 URLs found for task {task_id}")
            return jsonify({
                "success": False,
                "error": f"Task {task_id} has no output videos"
            }), 400
        
        print(f"🔍 [DEBUG] Starting upscaling in background thread for {len(task.output_s3_urls)} videos")
        # Start upscaling in background thread
        thread = threading.Thread(target=run_upscale_task, args=(task_id, task.output_s3_urls, app))
        thread.daemon = False
        thread.start()
        
        # Update task status
        print(f"🔍 [DEBUG] Marking task as upscaling started")
        task.mark_upscaling_started()
        task.save(mongo.db.lip_sync_tasks)
        
        logger.info(f"🚀 Started upscaling for task {task_id}")
        print(f"🔍 [DEBUG] Task updated in database")
        
        return jsonify({
            "success": True,
            "task_id": task_id,
            "message": "Upscaling started",
            "outputs_to_upscale": len(task.output_s3_urls)
        })
        
    except Exception as e:
        print(f"🔍 [DEBUG] Exception in upscale_task: {str(e)}")
        logger.error(f"❌ Failed to start upscaling for {task_id}: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/upscale/status/<task_id>', methods=['GET'])
def get_upscaling_status(task_id):
    """
    Get upscaling status for a task
    """
    print(f"🔍 [DEBUG] get_upscaling_status endpoint called with task_id: {task_id}")
    try:
        task_data = mongo.db.lip_sync_tasks.find_one({"task_id": task_id})
        print(f"🔍 [DEBUG] Task data found: {task_data is not None}")
        
        if not task_data:
            print(f"🔍 [DEBUG] Task {task_id} not found")
            return jsonify({"error": "Task not found"}), 404
        
        task = LipSyncTask.from_dict(task_data)
        print(f"🔍 [DEBUG] Upscaling status: {task.upscaling_status}")
        
        return jsonify({
            "success": True,
            "task_id": task_id,
            "upscaling_status": task.upscaling_status,
            "original_outputs": task.original_output_urls,
            "upscaled_outputs": task.upscaled_output_urls,
            "current_outputs": task.output_s3_urls,
            "error": task.upscaling_error,
            "started_at": task.upscaling_started_at,
            "completed_at": task.upscaling_completed_at
        })
        
    except Exception as e:
        print(f"🔍 [DEBUG] Exception in get_upscaling_status: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

def run_upscale_task(task_id, video_urls, app_context):
    """
    Run upscaling task inline with application context
    """
    print(f"🔍 [DEBUG] run_upscale_task called for task_id: {task_id} with {len(video_urls)} videos")
    
    # Set up logging for this task
    log_filename = f"logs/upscale_{task_id}.log"
    os.makedirs("logs", exist_ok=True)
    print(f"🔍 [DEBUG] Log file: {log_filename}")

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    task_logger = logging.getLogger(f"upscale_{task_id}")
    task_logger.setLevel(logging.INFO)
    task_logger.addHandler(file_handler)
    task_logger.propagate = False

    task_logger.info(f"🎬 Starting upscaling task for {task_id}")
    print(f"🔍 [DEBUG] Task logger initialized")

    with app_context.app_context():
        try:
            # Get task from database
            print(f"🔍 [DEBUG] Fetching task from MongoDB: {task_id}")
            task_data = mongo.db.lip_sync_tasks.find_one({"task_id": task_id})
            print(f"🔍 [DEBUG] Task found: {task_data is not None}")
            
            if not task_data:
                raise Exception(f"Task {task_id} not found")

            task = LipSyncTask.from_dict(task_data)
            print(f"🔍 [DEBUG] Task object created successfully")

            upscaled_urls = []

            for i, video_url in enumerate(video_urls):
                print(f"🔍 [DEBUG] Processing video {i+1}/{len(video_urls)}")
                task_logger.info(f"🔄 Upscaling video {i+1}/{len(video_urls)}: {video_url}")

                # Download video locally
                local_input_path = f"/tmp/{task_id}_input_{i}.mp4"
                local_gfpgan_path = f"/tmp/{task_id}_gfpgan_{i}.mp4"
                local_q17_path = f"/tmp/{task_id}_q17_{i}.mp4"

                print(f"🔍 [DEBUG] Input path: {local_input_path}")
                print(f"🔍 [DEBUG] GFPGAN path: {local_gfpgan_path}")
                print(f"🔍 [DEBUG] Q17 path: {local_q17_path}")

                # Download video
                print(f"🔍 [DEBUG] Downloading video...")
                download_video(video_url, local_input_path, task_logger)
                print(f"🔍 [DEBUG] Video downloaded successfully")

                # Run GFPGAN enhancement
                print(f"🔍 [DEBUG] Starting GFPGAN enhancement...")
                upscale_video_with_gfpgan(local_input_path, local_gfpgan_path, logger=task_logger)
                print(f"🔍 [DEBUG] GFPGAN enhancement completed")

                # Apply FFmpeg quality enhancement (CRF 17)
                print(f"🔍 [DEBUG] Applying FFmpeg quality enhancement (CRF 17)...")
                apply_ffmpeg_quality(local_gfpgan_path, local_q17_path, task_logger)
                print(f"🔍 [DEBUG] FFmpeg quality enhancement completed")

                # Generate audio sync variations
                print(f"🔍 [DEBUG] Generating audio sync variations...")
                sync_urls = apply_audio_sync_offsets(local_q17_path, task_id, i, task_logger)
                upscaled_urls.extend(sync_urls)
                print(f"🔍 [DEBUG] Generated {len(sync_urls)} sync variations")

                # Cleanup local files
                print(f"🔍 [DEBUG] Cleaning up local files...")
                cleanup_files([local_input_path, local_gfpgan_path, local_q17_path])

                task_logger.info(f"✅ Upscaled video {i+1} with {len(sync_urls)} sync variations")

            # Update task with upscaled URLs
            print(f"🔍 [DEBUG] Marking task as upscaling completed with {len(upscaled_urls)} URLs")
            task.mark_upscaling_completed(upscaled_urls)
            task.save(mongo.db.lip_sync_tasks)
            print(f"🔍 [DEBUG] Task saved to database")

            task_logger.info(f"🎉 Upscaling completed for task {task_id}")
            print(f"🔍 [DEBUG] Upscaling completed successfully")

        except Exception as e:
            print(f"🔍 [DEBUG] Exception in run_upscale_task: {str(e)}")
            task_logger.error(f"❌ Upscaling failed for {task_id}: {e}", exc_info=True)

            # Update task with failure
            try:
                task_data = mongo.db.lip_sync_tasks.find_one({"task_id": task_id})
                if task_data:
                    task = LipSyncTask.from_dict(task_data)
                    task.mark_upscaling_failed(str(e))
                    task.save(mongo.db.lip_sync_tasks)
                    print(f"🔍 [DEBUG] Task failure recorded in database")
            except Exception as db_error:
                print(f"🔍 [DEBUG] Failed to update task failure in DB: {str(db_error)}")
                task_logger.error(f"Failed to record failure in database: {db_error}")

def download_video(video_url, local_path, logger=None):
    """Download video from URL to local path"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    print(f"🔍 [DEBUG] download_video: URL={video_url}, Path={local_path}")
    try:
        print(f"🔍 [DEBUG] Sending GET request to {video_url}")
        response = requests.get(video_url, stream=True, timeout=60)
        response.raise_for_status()
        print(f"🔍 [DEBUG] Response status: {response.status_code}")

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"🔍 [DEBUG] File saved successfully. Size: {os.path.getsize(local_path)} bytes")
        logger.info(f"📥 Downloaded video to {local_path}")

    except Exception as e:
        print(f"🔍 [DEBUG] Download failed: {str(e)}")
        raise Exception(f"Failed to download video: {e}")

def upscale_video_with_gfpgan(input_path, output_path, upscale_factor=2, logger=None):
    """
    Run GFPGAN enhancement on video using enhance.py script with live log capture
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    print(f"🔍 [DEBUG] upscale_video_with_gfpgan: input={input_path}, output={output_path}, factor={upscale_factor}")
    try:
        enhance_script = "enhance.py"
        print(f"🔍 [DEBUG] Using enhance script: {enhance_script}")
        print(f"🔍 [DEBUG] Script exists: {os.path.exists(enhance_script)}")

        command = [
            "python", enhance_script,
            "--input", input_path,
            "--output", output_path,
            "--model", "gfpgan",
            "--upscale", str(upscale_factor),
            "--gpu-id", "0"
        ]

        logger.info(f"⚙️ Running GFPGAN: {' '.join(command)}")
        print(f"🔍 [DEBUG] Running command: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        print(f"🔍 [DEBUG] Process started with PID: {process.pid}")
        
        # Read output line by line in real time
        try:
            for line in process.stdout:
                line = line.rstrip('\n')
                if line:
                    print(f"🔍 [GFPGAN] {line}")
                    logger.info(f"[enhance.py] {line}")
        except:
            pass
        
        process.wait(timeout=3600)
        
        print(f"🔍 [DEBUG] Process completed with return code: {process.returncode}")

        if process.returncode != 0:
            raise Exception(f"GFPGAN enhancement failed with return code {process.returncode}")

        print(f"🔍 [DEBUG] Checking if output file exists: {output_path}")
        if not os.path.exists(output_path):
            raise Exception("Upscaled video not created")

        print(f"🔍 [DEBUG] Output file exists. Size: {os.path.getsize(output_path)} bytes")
        logger.info(f"✅ GFPGAN enhancement completed: {output_path}")

    except subprocess.TimeoutExpired:
        print(f"🔍 [DEBUG] Process timeout, killing process...")
        process.kill()
        logger.error("Process timeout")
        raise Exception("GFPGAN enhancement timed out")
    except Exception as e:
        print(f"🔍 [DEBUG] Exception in upscale_video_with_gfpgan: {str(e)}")
        logger.error(f"❌ GFPGAN enhancement failed: {e}", exc_info=True)
        raise Exception(f"GFPGAN enhancement failed: {e}")

def apply_ffmpeg_quality(input_path, output_path, logger=None):
    """
    Apply FFmpeg quality enhancement (CRF 17, libx264, yuv420p)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    print(f"🔍 [DEBUG] apply_ffmpeg_quality: input={input_path}, output={output_path}")
    try:
        command = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-crf", "17",
            "-preset", "slow",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "160k",
            output_path
        ]

        logger.info(f"⚙️ Applying FFmpeg quality: CRF 17")
        print(f"🔍 [DEBUG] Running FFmpeg quality command")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(timeout=1800)
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg quality enhancement failed: {stderr}")

        if not os.path.exists(output_path):
            raise Exception("Quality-enhanced video not created")

        print(f"🔍 [DEBUG] Quality enhancement completed. Size: {os.path.getsize(output_path)} bytes")
        logger.info(f"✅ FFmpeg quality enhancement (CRF 17) completed")

    except subprocess.TimeoutExpired:
        print(f"🔍 [DEBUG] FFmpeg process timeout, killing...")
        process.kill()
        raise Exception("FFmpeg quality enhancement timed out")
    except Exception as e:
        print(f"🔍 [DEBUG] Exception in apply_ffmpeg_quality: {str(e)}")
        logger.error(f"❌ FFmpeg quality enhancement failed: {e}")
        raise Exception(f"FFmpeg quality enhancement failed: {e}")

def apply_audio_sync_offsets(input_path, task_id, video_index, logger=None):
    """
    Generate audio sync variations with different offsets
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    print(f"🔍 [DEBUG] apply_audio_sync_offsets: input={input_path}, task_id={task_id}")
    
    upscaled_urls = []
    
    for offset in AUDIO_SYNC_OFFSETS:
        try:
            # Format offset for filename (e.g., -0.12 -> m0p12s, 0.04 -> 0p04s)
            offset_str = f"{offset:.2f}".replace("-", "m").replace(".", "p") + "s"
            local_sync_path = f"/tmp/{task_id}_sync_{video_index}_{offset_str}.mp4"
            
            print(f"🔍 [DEBUG] Processing offset {offset}s: {local_sync_path}")
            
            command = [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-itsoffset", str(offset),
                "-i", input_path,
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "160k",
                local_sync_path
            ]

            logger.info(f"⚙️ Creating audio sync variation: offset={offset}s")
            print(f"🔍 [DEBUG] Running FFmpeg sync command for offset {offset}")

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stdout, stderr = process.communicate(timeout=600)
            
            if process.returncode != 0:
                logger.warning(f"⚠️ Failed to create sync variation {offset}s: {stderr}")
                print(f"🔍 [DEBUG] Sync variation failed, continuing...")
                continue

            if not os.path.exists(local_sync_path):
                logger.warning(f"⚠️ Sync file not created for offset {offset}s")
                continue

            # Upload to S3
            s3_key = f"upscaled_outputs/{task_id}/model_{video_index}_sync_{offset_str}.mp4"
            upscaled_url = upload_to_s3(local_sync_path, s3_key, logger)
            upscaled_urls.append(upscaled_url)
            
            print(f"🔍 [DEBUG] Uploaded sync variation: {upscaled_url}")
            logger.info(f"✅ Audio sync offset {offset}s uploaded: {upscaled_url}")

            # Cleanup sync file
            try:
                os.remove(local_sync_path)
                print(f"🔍 [DEBUG] Cleaned up: {local_sync_path}")
            except:
                pass

        except subprocess.TimeoutExpired:
            print(f"🔍 [DEBUG] Timeout for offset {offset}, skipping...")
            logger.warning(f"⚠️ Audio sync offset {offset}s timed out")
        except Exception as e:
            print(f"🔍 [DEBUG] Error processing offset {offset}: {str(e)}")
            logger.warning(f"⚠️ Error processing offset {offset}s: {e}")

    print(f"🔍 [DEBUG] Generated {len(upscaled_urls)} sync variations")
    return upscaled_urls

def upload_to_s3(local_path, s3_key, logger=None):
    """Upload file to S3 and return public URL"""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    print(f"🔍 [DEBUG] upload_to_s3: local_path={local_path}, s3_key={s3_key}")
    
    try:
        import boto3
        
        # Get AWS credentials from environment
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        bucket_name = os.getenv("S3_BUCKET_NAME")
        
        print(f"🔍 [DEBUG] AWS Config - Region: {aws_region}, Bucket: {bucket_name}")
        
        if not all([aws_access_key, aws_secret_key, bucket_name]):
            raise Exception("Missing AWS credentials in environment variables")
        
        # Create S3 client
        print(f"🔍 [DEBUG] Creating S3 client...")
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        # Check if file exists
        if not os.path.exists(local_path):
            raise Exception(f"Local file does not exist: {local_path}")
        
        file_size = os.path.getsize(local_path)
        print(f"🔍 [DEBUG] File size: {file_size / (1024**2):.2f} MB")
        logger.info(f"📤 Uploading file to S3: {s3_key}")
        print(f"🔍 [DEBUG] Uploading to S3: s3://{bucket_name}/{s3_key}")
        
        # Upload file
        s3_client.upload_file(
            local_path,
            bucket_name,
            s3_key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        
        print(f"🔍 [DEBUG] Upload completed successfully")
        logger.info(f"✅ File uploaded successfully")
        
        # Generate public URL
        url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_key}"
        print(f"🔍 [DEBUG] Generated S3 URL: {url}")
        logger.info(f"📍 S3 URL: {url}")
        
        return url
    
    except Exception as e:
        print(f"🔍 [DEBUG] S3 upload error: {str(e)}")
        logger.error(f"❌ Failed to upload to S3: {e}", exc_info=True)
        raise Exception(f"S3 upload failed: {e}")

def cleanup_files(file_paths):
    """Clean up local files"""
    print(f"🔍 [DEBUG] cleanup_files: {file_paths}")
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🔍 [DEBUG] Deleted: {file_path}")
        except Exception as e:
            print(f"🔍 [DEBUG] Failed to delete {file_path}: {str(e)}")
            logger.warning(f"Failed to cleanup {file_path}: {e}")

def save_upscaled_urls_to_db(task_id, upscaled_urls, task_logger=None):
    """Save upscaled URLs to database"""
    if task_logger is None:
        task_logger = logging.getLogger(__name__)
    
    try:
        print(f"🔍 [DEBUG] Saving {len(upscaled_urls)} upscaled URLs to database")
        task_logger.info(f"💾 Saving {len(upscaled_urls)} upscaled URLs to database")
        
        # Fetch task from database
        task_data = mongo.db.lip_sync_tasks.find_one({"task_id": task_id})
        if not task_data:
            raise Exception(f"Task {task_id} not found in database")
        
        task = LipSyncTask.from_dict(task_data)
        
        # Mark upscaling as completed with the URLs
        task.mark_upscaling_completed(upscaled_urls)
        task.save(mongo.db.lip_sync_tasks)
        
        print(f"🔍 [DEBUG] URLs saved successfully to database")
        task_logger.info(f"✅ {len(upscaled_urls)} URLs saved to database")
        
        return True
    
    except Exception as e:
        print(f"🔍 [DEBUG] Failed to save URLs to database: {str(e)}")
        task_logger.error(f"❌ Failed to save URLs to database: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    print("🔍 [DEBUG] Starting Flask app...")
    print("\n📦 SETUP REQUIRED:")
    print("   Make sure you have boto3 installed:")
    print("   pip install boto3")
    print("\n🔐 ENVIRONMENT VARIABLES REQUIRED:")
    print("   AWS_ACCESS_KEY_ID")
    print("   AWS_SECRET_ACCESS_KEY")
    print("   AWS_REGION (default: us-east-1)")
    print("   S3_BUCKET_NAME")
    print("\n" + "="*80)
    app.run(host='0.0.0.0', port=UPSCALING_SERVICE_PORT, debug=False)