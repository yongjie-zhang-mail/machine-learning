import os
import boto3
import requests
from moviepy.editor import VideoFileClip

def download_video(url, output_path):
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

def upload_to_s3(file_path, bucket_name, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket_name, s3_key)

def process_video(url, start_time, bucket_name, s3_key):
    video_path = 'temp_video.mp4'
    clip_path = 'temp_clip.mp4'
    
    # 下载视频
    download_video(url, video_path)
    
    # 截取视频片段
    with VideoFileClip(video_path) as video:
        clip = video.subclip(start_time, start_time + 15)
        clip.write_videofile(clip_path, codec='libx264')
    
    # 上传到 S3
    upload_to_s3(clip_path, bucket_name, s3_key)
    
    # 删除临时文件
    os.remove(video_path)
    os.remove(clip_path)

# 示例用法
if __name__ == "__main__":
    video_url = "https://example.com/path/to/video.mp4"
    start_time = 10  # 以秒为单位
    bucket_name = "your-s3-bucket-name"
    s3_key = "path/to/uploaded_clip.mp4"
    
    process_video(video_url, start_time, bucket_name, s3_key)
