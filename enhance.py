"""
Video Enhancement Script for Wav2Lip Output
Enhances blurry lip-synced videos using face restoration models
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import os

# You'll need to install these packages:
# pip install opencv-python
# pip install basicsr
# pip install facexlib
# pip install gfpgan
# pip install realesrgan

from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import torch


class VideoEnhancer:
    def __init__(self, model_type='gfpgan', upscale=2, gpu_id=0):
        """
        Initialize the video enhancer
        
        Args:
            model_type: 'gfpgan', 'codeformer', or 'realesrgan'
            upscale: upscale factor (1, 2, or 4)
            gpu_id: GPU device ID (default: 0)
        """
        self.model_type = model_type
        self.upscale = upscale
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.restorer = None
        
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(gpu_id)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the enhancement model"""
        print(f"Loading {self.model_type} model...")
        
        # Use FP16 (half precision) only if CUDA is available
        use_half = torch.cuda.is_available()
        
        if self.model_type == 'gfpgan':
            # GFPGAN is excellent for face restoration
            model_path = 'weights/GFPGANv1.4.pth'
            
            # Create RealESRGAN upsampler as background enhancer
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                          num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(
                scale=2,
                model_path='weights/RealESRGAN_x2plus.pth',
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=use_half,
                device=self.device
            )
            
            self.restorer = GFPGANer(
                model_path=model_path,
                upscale=self.upscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=bg_upsampler,
                device=self.device
            )
        
        elif self.model_type == 'realesrgan':
            # Real-ESRGAN for general enhancement
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                          num_block=23, num_grow_ch=32, scale=self.upscale)
            
            model_path = f'weights/RealESRGAN_x{self.upscale}plus.pth'
            
            self.restorer = RealESRGANer(
                scale=self.upscale,
                model_path=model_path,
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=use_half,
                device=self.device
            )
        
        print("Model loaded successfully!")
    
    def enhance_frame(self, frame):
        """
        Enhance a single frame
        
        Args:
            frame: input frame (numpy array)
        
        Returns:
            enhanced frame
        """
        if self.model_type == 'gfpgan':
            # GFPGAN returns: cropped_faces, restored_faces, restored_img
            _, _, output = self.restorer.enhance(
                frame, 
                has_aligned=False, 
                only_center_face=False, 
                paste_back=True,
                weight=0.5  # Adjust blend weight (0-1)
            )
        else:
            # Real-ESRGAN
            output, _ = self.restorer.enhance(frame, outscale=self.upscale)
        
        return output
    
    def enhance_video(self, input_path, output_path, start_frame=0, end_frame=None):
        """
        Enhance entire video
        
        Args:
            input_path: path to input video
            output_path: path to save enhanced video
            start_frame: frame to start from (default: 0)
            end_frame: frame to end at (default: None = end of video)
        """
        print(f"Enhancing video: {input_path}")
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if end_frame is None:
            end_frame = total_frames
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        print(f"Processing frames {start_frame} to {end_frame}")
        
        # Prepare output video writer
        output_width = width * self.upscale
        output_height = height * self.upscale
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, 
                            (output_width, output_height))
        
        # Process frames
        frame_count = 0
        processed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count < start_frame:
                frame_count += 1
                continue
            
            if frame_count >= end_frame:
                break
            
            # Enhance frame
            enhanced_frame = self.enhance_frame(frame)
            
            # Write enhanced frame
            out.write(enhanced_frame)
            
            processed_count += 1
            if processed_count % 30 == 0:
                print(f"Processed {processed_count}/{end_frame - start_frame} frames "
                      f"({100 * processed_count / (end_frame - start_frame):.1f}%)")
            
            frame_count += 1
        
        # Release resources
        cap.release()
        out.release()
        
        print(f"Enhanced video saved to: {output_path}")
        print(f"Total frames processed: {processed_count}")
    
    def enhance_with_audio(self, input_video, output_video, audio_path=None):
        """
        Enhance video and preserve/add audio
        
        Args:
            input_video: path to input video
            output_video: path to save enhanced video
            audio_path: optional separate audio file (uses input video audio if None)
        """
        # First enhance the video
        temp_output = output_video.replace('.mp4', '_no_audio.mp4')
        self.enhance_video(input_video, temp_output)
        
        # Then add audio using ffmpeg
        if audio_path is None:
            audio_source = input_video
        else:
            audio_source = audio_path
        
        print("Adding audio to enhanced video...")
        cmd = f'ffmpeg -i "{temp_output}" -i "{audio_source}" -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 "{output_video}" -y'
        os.system(cmd)
        
        # Clean up temp file
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        print(f"Final video with audio saved to: {output_video}")


def download_models():
    """Download required model weights if not present"""
    import requests
    os.makedirs('weights', exist_ok=True)

    models = {
        'GFPGANv1.4.pth': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
        'RealESRGAN_x2plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        'RealESRGAN_x4plus.pth': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    }

    for model_name, url in models.items():
        model_path = f'weights/{model_name}'
        if not os.path.exists(model_path):
            print(f"Downloading {model_name}...")
            response = requests.get(url)
            with open(model_path, 'wb') as f:
                f.write(response.content)
            print(f"✓ Downloaded {model_name}")
        else:
            print(f"✓ {model_name} already exists")

    print("\nAll models downloaded!")


def main():
    parser = argparse.ArgumentParser(description='Enhance Wav2Lip output video')
    parser.add_argument('--input', '-i', help='Input video path')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--model', '-m', default='gfpgan', 
                       choices=['gfpgan', 'realesrgan'],
                       help='Enhancement model to use')
    parser.add_argument('--upscale', '-u', type=int, default=2,
                       choices=[1, 2, 4], help='Upscale factor')
    parser.add_argument('--audio', '-a', default=None,
                       help='Optional audio file to add (uses input video audio if not specified)')
    parser.add_argument('--download-models', action='store_true',
                       help='Download model weights before processing')
    parser.add_argument('--start-frame', type=int, default=0,
                       help='Start frame for processing')
    parser.add_argument('--end-frame', type=int, default=None,
                       help='End frame for processing')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Download models if requested
    if args.download_models:
        download_models()
        print("\n✓ Models downloaded successfully!")
        if not args.input:
            print("\nTo enhance a video, run:")
            print("python enhance.py -i input_video.mp4 -o output_video.mp4")
            return
    
    # Check if input/output are provided
    if not args.input or not args.output:
        parser.error("--input and --output are required for video enhancement")
    
    # Create enhancer
    enhancer = VideoEnhancer(model_type=args.model, upscale=args.upscale, gpu_id=args.gpu_id)
    
    # Enhance video
    if args.audio is not None or os.path.exists(args.input):
        enhancer.enhance_with_audio(args.input, args.output, args.audio)
    else:
        enhancer.enhance_video(args.input, args.output, args.start_frame, args.end_frame)


if __name__ == '__main__':
    main()
