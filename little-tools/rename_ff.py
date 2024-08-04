import os


class RenameFF:

    def __init__(self):
        pass

    def rename_folders(self, path):
        """
        Renames folders in the specified path by removing the characters before the '@' symbol in the folder name.

        Args:
            path (str): The path to the directory containing the folders to be renamed.

        Returns:
            None
        """
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


    def rename_folders_current_dir_recursive(self):
        """
        Renames folders in the current directory and all its subdirectories by removing the characters before the '@' symbol in the folder name.

        Args:
            None

        Returns:
            None
        """
        # 遍历当前目录
        for root, dirs, files in os.walk(os.getcwd()):
            for dirname in dirs:
                if '@' in dirname:
                    # 提取 '@' 之后的部分
                    new_dirname = dirname.split('@', 1)[1]
                    
                    # 构建完整的路径
                    old_dirpath = os.path.join(root, dirname)
                    new_dirpath = os.path.join(root, new_dirname)
                    
                    # 重命名文件夹
                    os.rename(old_dirpath, new_dirpath)
                    print(f"Renamed '{dirname}' to '{new_dirname}'")



    def rename_folders_current_dir_recursive2(self):
        """
        Renames folders in the current directory and all its subdirectories by removing the characters before the '@' symbol in the folder name.

        Args:
            None

        Returns:
            None
        """
        # 遍历当前目录
        for root, dirs, files in os.walk(os.getcwd()):
            # 处理当前层级的所有目录
            dirs[:] = [d for d in dirs if not ('@' in d and d.split('@', 1)[1] != d)]
            
            for dirname in dirs[:]:  # 使用一个拷贝来安全地遍历和修改
                if '@' in dirname:
                    # 提取 '@' 之后的部分
                    new_dirname = dirname.split('@', 1)[1]
                    
                    # 构建完整的路径
                    old_dirpath = os.path.join(root, dirname)
                    new_dirpath = os.path.join(root, new_dirname)
                    
                    # 重命名文件夹
                    os.rename(old_dirpath, new_dirpath)
                    print(f"Renamed '{dirname}' to '{new_dirname}'")


    def rename_files(self, path):
        """
        Renames files in the specified directory by removing the characters before the '@' symbol in the filename.

        Args:
            path (str): The path to the directory containing the files to be renamed.

        Returns:
            None
        """
        # 遍历指定目录
        for filename in os.listdir(path):
            if '@' in filename:
                # 提取 '@' 之后的部分
                new_filename = filename.split('@', 1)[1]
                
                # 构建完整的路径
                old_filepath = os.path.join(path, filename)
                new_filepath = os.path.join(path, new_filename)
                
                # 重命名文件
                os.rename(old_filepath, new_filepath)
                print(f"Renamed '{filename}' to '{new_filename}'")


    def rename_files_current_dir(self):
        """
        Renames files in the current directory by removing the characters before the '@' symbol in the filename.

        Args:
            None

        Returns:
            None
        """
        # 遍历当前目录
        for filename in os.listdir():
            if '@' in filename:
                # 提取 '@' 之后的部分
                new_filename = filename.split('@', 1)[1]
                
                # 构建完整的路径
                old_filepath = os.path.join(os.getcwd(), filename)
                new_filepath = os.path.join(os.getcwd(), new_filename)
                
                # 重命名文件
                os.rename(old_filepath, new_filepath)
                print(f"Renamed '{filename}' to '{new_filename}'")

    
    def rename_files_current_dir_recursive(self):
        """
        Renames files in the current directory and all its subdirectories by removing the characters before the '@' symbol in the filename.

        Args:
            None

        Returns:
            None
        """
        # 遍历当前目录
        for root, dirs, files in os.walk(os.getcwd()):
            for filename in files:
                if '@' in filename:
                    # 提取 '@' 之后的部分
                    new_filename = filename.split('@', 1)[1]
                    
                    # 构建完整的路径
                    old_filepath = os.path.join(root, filename)
                    new_filepath = os.path.join(root, new_filename)
                    
                    # 重命名文件
                    os.rename(old_filepath, new_filepath)
                    print(f"Renamed '{filename}' to '{new_filename}'")


if __name__ == '__main__':
    rff = RenameFF()
    # 调用函数，传入你想要遍历的目录路径
    # rff.rename_folders('/path/to/your/directory')  
    # rff.rename_folders('D:\download8')
    # 调用函数
    # rff.rename_files(r'C:\download1\temp2')
    # rff.rename_files_current_dir()

    rff.rename_folders_current_dir_recursive()
    rff.rename_files_current_dir_recursive()
    
    


