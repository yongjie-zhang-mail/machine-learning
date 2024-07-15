import os


class RenameFolder:

    def __init__(self):
        pass

    def rename_folders(self, path):
        # 遍历指定路径下的所有文件和文件夹
        for filename in os.listdir(path):
            # 检查是否为文件夹
            if os.path.isdir(os.path.join(path, filename)):
                # 查找 @ 符号
                at_pos = filename.find('@')
                if at_pos != -1:
                    # 提取 @ 之后的部分
                    new_name = filename[at_pos + 1:]
                    # 构建新的路径
                    new_path = os.path.join(path, new_name)
                    # 重命名文件夹
                    os.rename(os.path.join(path, filename), new_path)
                    print(f"Renamed '{filename}' to '{new_name}'")


if __name__ == '__main__':
    rf = RenameFolder()
    # 调用函数，传入你想要遍历的目录路径
    # rf.rename_folders('/path/to/your/directory')  
    rf.rename_folders('D:\download8')


    
    


