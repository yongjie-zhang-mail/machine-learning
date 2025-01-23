import requests
from moviepy.editor import VideoFileClip

def download_video(url, output_path='downloaded_video.mp4'):
    response = requests.get(url, stream=True)
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return output_path

def extract_clip_from_url(url, start_time, duration=15, output_path='output_clip.mp4'):
    video_path = download_video(url)
    video = VideoFileClip(video_path)
    print(f"Video duration: {video.duration}")
    
    clip = video.subclip(start_time, start_time + duration)
    clip.write_videofile(output_path, codec='libx264')

def extract_clip(video_url, start_time, duration=15, output_path='output_clip.mp4'):
    # 加载视频文件
    video = VideoFileClip(video_url)
    
    # 截取视频片段
    clip = video.subclip(start_time, start_time + duration)
    
    # 保存视频片段
    clip.write_videofile(output_path, codec='libx264')

if __name__ == "__main__":
    # 输入视频的URL
    video_url = "https://leforge.lenovo.com/res/upload/leforge/0172b4914a9247d7a037c99e52f9e0d7.mp4"
    # 输入开始时间（秒）
    start_time = 10
    extract_clip_from_url(video_url, start_time)
    
    # 示例用法
    video_url = "path/to/your/video.mp4"  # 替换为实际视频文件路径
    start_time = 10  # 替换为实际开始时间（秒）
    extract_clip(video_url, start_time)




