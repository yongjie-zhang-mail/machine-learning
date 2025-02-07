
import os
import requests
# 引入的包的版本的问题，导致无法使用 VideoFileClip
# copilot 生成的代码，使用的是 moviepy==1.0.3 老的版本，默认安装的 2.* 版本不支持
# 可以通过 pip install moviepy==1.0.3 安装老版本
# 具体信息，可以查看 github 或 pypi 的文档
from moviepy.editor import VideoFileClip



def extract_clip(video_path, start_time, duration=15, clip_path='output_clip.mp4'):
    """
    描述:
        从给定视频中截取指定开始时刻的 15 秒视频片段，并将剪辑保存于本地。
    参数:
        video_path (str): 视频的文件路径。
        start_time (int 或 float): 截取剪辑的起始时间（单位：秒）。
        duration (int, 可选): 截取的剪辑时长，默认为 15 秒。
        clip_path (str, 可选): 输出剪辑的保存路径，默认为 'output_clip.mp4'。
    返回:
        clip_path (str): 返回视频剪辑保存的文件路径。
    示例:
        >>> extract_clip("input_video.mp4", 10)
        # 此示例将从 "input_video.mp4" 的第 10 秒开始截取 15 秒的视频片段，并保存为 "output_clip.mp4"。
    """

    # 加载视频
    video = VideoFileClip(video_path)    
    # 截取视频片段
    clip = video.subclip(start_time, start_time + duration)    
    # 保存视频片段
    clip.write_videofile(clip_path, codec='libx264')
    # 返回视频剪辑保存的文件路径
    return clip_path



def download_video(video_url, download_path='downloaded_video.mp4'):
    """
    下载视频函数。

    从指定的 URL 下载视频文件，并将下载的数据写入到指定的本地文件。

    参数:
        video_url (str): 指定要下载视频的 URL 地址。
        download_path (str, optional): 视频保存的文件路径，默认为 'downloaded_video.mp4'。

    返回:
        download_path (str): 返回视频下载后保存的文件路径。
    """
    
    # 判断 downloaded_video.mp4 是否存在，如果存在则直接返回
    if os.path.exists(download_path):
        return download_path

    response = requests.get(video_url, stream=True)
    with open(download_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return download_path


# 示例用法
if __name__ == "__main__":    

    # 下载原始视频
    video_url = ""  
    download_path = download_video(video_url=video_url)
    print(f"原始视频下载路径：{download_path}")

    # # 截取视频片段    
    start_time = 30  
    clip_path = extract_clip(video_path=download_path, start_time=start_time)
    print(f"截取的视频片段路径：{clip_path}")

