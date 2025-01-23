
# 引入的包的版本的问题，导致无法使用 VideoFileClip
# copilot 生成的代码，使用的是 moviepy==1.0.3 老的版本，默认安装的 2.* 版本不支持
# 可以通过 pip install moviepy==1.0.3 安装老版本
# 具体信息，可以查看 github 或 pypi 的文档
from moviepy.editor import VideoFileClip
# from moviepy import VideoFileClip

def extract_clip(video_url, start_time, duration=15, output_path='output_clip.mp4'):
    """
    从给定的视频 URL 截取指定开始时间的 15 秒视频片段，并保存到本地。

    :param video_url: 视频的 URL
    :param start_time: 开始时间（秒）
    :param duration: 截取时长（秒），默认为 15 秒
    :param output_path: 输出文件路径，默认为 'output_clip.mp4'
    """
    # 加载视频
    video = VideoFileClip(video_url)
    
    # 截取视频片段
    clip = video.subclip(start_time, start_time + duration)
    
    
    # 保存视频片段
    clip.write_videofile(output_path, codec='libx264')

# 示例用法
if __name__ == "__main__":
    # 替换为实际视频路径
    # video_url = "https://leforge.lenovo.com/res/upload/leforge/0172b4914a9247d7a037c99e52f9e0d7.mp4"  
    video_url = "/source/lab/machine-learning/downloaded_video.mp4"
    # 替换为实际开始时间
    start_time = 30  
    extract_clip(video_url, start_time)
