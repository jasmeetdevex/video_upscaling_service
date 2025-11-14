from flask import Flask, request, jsonify
from flask_cors import CORS
from extensions import mongo, init_mongo
from models.lipSyncTask import LipSyncTask
import requests
import subprocess
import os
import logging
from datetime import datetime
import multiprocessing

print("🔍 [DEBUG] app.py starting...")

app = Flask(__name__)
CORS(app)
init_mongo(app)

print("🔍 [DEBUG] Flask app initialized")
print(f"🔍 [DEBUG] MongoDB initialized: {mongo}")

logger = logging.getLogger(__name__)

# Upscaling service configuration
UPSCALING_SERVICE_PORT = 5002

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
        
        # Check if task is completed and has outputs
        # if task.status != "completed":
        #     print(f"🔍 [DEBUG] Task status is '{task.status}', expected 'completed'")
        #     return jsonify({
        #         "success": False,
        #         "error": f"Task {task_id} is not completed yet (status: {task.status})"
        #     }), 400
        
        print(f"🔍 [DEBUG] Task status is completed. Output URLs count: {len(task.output_s3_urls)}")
        
        if not task.output_s3_urls:
            print(f"🔍 [DEBUG] No output S3 URLs found for task {task_id}")
            return jsonify({
                "success": False,
                "error": f"Task {task_id} has no output videos"
            }), 400
        
        print(f"🔍 [DEBUG] Starting upscaling subprocess for {len(task.output_s3_urls)} videos")
        # Start upscaling in subprocess
        run_upscale_task(task_id, task.output_s3_urls)
        # process = multiprocessing.Process(target=run_upscale_task, args=(task_id, task.output_s3_urls))
        # process.start()
        
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

def run_upscale_task(task_id, video_urls):
    """
    Run upscaling task in subprocess with logging to file
    """
    print(f"🔍 [DEBUG] run_upscale_task called for task_id: {task_id} with {len(video_urls)} videos")
    
    # Set up logging for subprocess
    log_filename = f"logs/upscale_{task_id}.log"
    os.makedirs("logs", exist_ok=True)
    print(f"🔍 [DEBUG] Log file: {log_filename}")

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    subprocess_logger = logging.getLogger(f"upscale_{task_id}")
    subprocess_logger.setLevel(logging.INFO)
    subprocess_logger.addHandler(file_handler)
    subprocess_logger.propagate = False

    # Initialize MongoDB in subprocess
    from extensions import mongo, init_mongo
    print(f"🔍 [DEBUG] Initializing MongoDB in subprocess")
    init_mongo(app)

    subprocess_logger.info(f"🎬 Starting upscaling task for {task_id}")
    print(f"🔍 [DEBUG] Subprocess logger initialized")

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
            subprocess_logger.info(f"🔄 Upscaling video {i+1}/{len(video_urls)}: {video_url}")

            # Download video locally
            local_input_path = f"/tmp/{task_id}_input_{i}.mp4"
            local_output_path = f"/tmp/{task_id}_upscaled_{i}.mp4"

            print(f"🔍 [DEBUG] Input path: {local_input_path}")
            print(f"🔍 [DEBUG] Output path: {local_output_path}")

            # Download video
            print(f"🔍 [DEBUG] Downloading video...")
            download_video(video_url, local_input_path, subprocess_logger)
            print(f"🔍 [DEBUG] Video downloaded successfully")

            # Run GFPGAN enhancement
            print(f"🔍 [DEBUG] Starting GFPGAN enhancement...")
            upscale_video_with_gfpgan(local_input_path, local_output_path, logger=subprocess_logger)
            print(f"🔍 [DEBUG] GFPGAN enhancement completed")

            # Upload upscaled video to S3
            print(f"🔍 [DEBUG] Uploading to S3...")
            upscaled_s3_key = f"upscaled_outputs/{task_id}/model_{i}.mp4"
            upscaled_url = upload_to_s3(local_output_path, upscaled_s3_key)
            upscaled_urls.append(upscaled_url)
            print(f"🔍 [DEBUG] Uploaded to S3: {upscaled_url}")

            # Cleanup local files
            print(f"🔍 [DEBUG] Cleaning up local files...")
            cleanup_files([local_input_path, local_output_path])

            subprocess_logger.info(f"✅ Upscaled video {i+1}: {upscaled_url}")

        # Update task with upscaled URLs
        print(f"🔍 [DEBUG] Marking task as upscaling completed with {len(upscaled_urls)} URLs")
        task.mark_upscaling_completed(upscaled_urls)
        task.save(mongo.db.lip_sync_tasks)
        print(f"🔍 [DEBUG] Task saved to database")

        subprocess_logger.info(f"🎉 Upscaling completed for task {task_id}")
        print(f"🔍 [DEBUG] Upscaling completed successfully")

    except Exception as e:
        print(f"🔍 [DEBUG] Exception in run_upscale_task: {str(e)}")
        subprocess_logger.error(f"❌ Upscaling failed for {task_id}: {e}")

        # Update task with failure
        task_data = mongo.db.lip_sync_tasks.find_one({"task_id": task_id})
        if task_data:
            task = LipSyncTask.from_dict(task_data)
            task.mark_upscaling_failed(str(e))
            task.save(mongo.db.lip_sync_tasks)
            print(f"🔍 [DEBUG] Task failure recorded in database")

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
    Run GFPGAN enhancement on video using subprocess
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    print(f"🔍 [DEBUG] upscale_video_with_gfpgan: input={input_path}, output={output_path}, factor={upscale_factor}")
    try:
        # Using your enhance.py script
        enhance_script = "enhance.py"  # Path to your enhance script
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

        logger.info(f"⚙️ Running: {' '.join(command)}")
        print(f"🔍 [DEBUG] Running command: {' '.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        print(f"🔍 [DEBUG] Process started with PID: {process.pid}")
        stdout, stderr = process.communicate(timeout=3600)  # 1 hour timeout
        
        print(f"🔍 [DEBUG] Process completed with return code: {process.returncode}")
        print(f"🔍 [DEBUG] STDOUT: {stdout[:500]}")
        print(f"🔍 [DEBUG] STDERR: {stderr[:500]}")

        if process.returncode != 0:
            raise Exception(f"GFPGAN failed: {stderr}")

        print(f"🔍 [DEBUG] Checking if output file exists: {output_path}")
        if not os.path.exists(output_path):
            raise Exception("Upscaled video not created")

        print(f"🔍 [DEBUG] Output file exists. Size: {os.path.getsize(output_path)} bytes")
        logger.info(f"✅ GFPGAN enhancement completed: {output_path}")

    except subprocess.TimeoutExpired:
        print(f"🔍 [DEBUG] Process timeout, killing process...")
        process.kill()
        raise Exception("GFPGAN enhancement timed out")
    except Exception as e:
        print(f"🔍 [DEBUG] Exception in upscale_video_with_gfpgan: {str(e)}")
        raise Exception(f"GFPGAN enhancement failed: {e}")

def upload_to_s3(local_path, s3_key):
    """Upload file to S3 and return URL"""
    print(f"🔍 [DEBUG] upload_to_s3: local_path={local_path}, s3_key={s3_key}")
    
    # Use your existing S3 upload logic
    # This should match your main service's upload function
    bucket_name = os.getenv("S3_BUCKET_NAME")
    print(f"🔍 [DEBUG] S3 bucket name: {bucket_name}")
    
    # Your existing upload logic here
    # s3.upload_file(local_path, bucket_name, s3_key)
    # return f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
    
    # For now, return a placeholder
    result = f"s3://{bucket_name}/{s3_key}"
    print(f"🔍 [DEBUG] Returning S3 URL: {result}")
    return result

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

if __name__ == '__main__':
    print("🔍 [DEBUG] Starting Flask app...")
    app.run(host='0.0.0.0', port=UPSCALING_SERVICE_PORT, debug=False)