import os
import re
import shutil
import sys

# 默认用法: 把脚本放进待处理目录执行 python move_ff.py
#   递归移平子文件夹 + 去文件名方括号 + 删空目录 + 根目录文件改名
# 其它用法: python move_ff.py [目录] [move|rename] [-r] [-d] [--no-recursive]


def remove_brackets(name):
    """
    去掉文件名中的方括号，并在原 ']' 的位置补上 ' - ' 分隔符。

    规则：按方括号把文件名切成若干段，压缩每段空白、丢弃空段后用 ' - ' 连接。

    示例:
        '[Blacked] Hime Marie - Gym Body.mp4' -> 'Blacked - Hime Marie - Gym Body.mp4'
        'Title [1080p].mp4'                   -> 'Title - 1080p.mp4'
        '[A][B] Name.mp4'                     -> 'A - B - Name.mp4'

    参数:
        name (str): 文件名（可带扩展名）

    返回:
        str: 去掉方括号后的文件名；名字全是方括号时原样返回
    """
    if '[' not in name and ']' not in name:
        return name

    base, ext = os.path.splitext(name)
    parts = (' '.join(p.split()) for p in re.split(r'[\[\]]', base))
    new_base = ' - '.join(p for p in parts if p)

    return new_base + ext if new_base else name


def _unique_path(root_dir, filename):
    """
    在 root_dir 下为 filename 生成一个不冲突的路径（冲突时追加 _1、_2 ...）。
    """
    base, ext = os.path.splitext(filename)
    dest_path = os.path.join(root_dir, filename)
    counter = 0
    while os.path.exists(dest_path):
        counter += 1
        dest_path = os.path.join(root_dir, f"{base}_{counter}{ext}")
    return dest_path


def _resolve_dir(path):
    """
    返回目录的绝对路径；目录不存在时打印错误并返回 None。
    """
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        print(f"错误：目录不存在 - {path}")
        return None
    return path


def _rename_in_dir(root_dir, recursive=False, rename_dirs=False):
    """去掉目录下文件名（rename_dirs 时含目录名）中的方括号；recursive 时含各级子目录。"""
    for current_root, dirs, files in os.walk(root_dir):
        for filename in files:
            new_name = remove_brackets(filename)
            if new_name == filename:
                continue
            new_path = _unique_path(current_root, new_name)
            os.rename(os.path.join(current_root, filename), new_path)
            print(f"  重命名文件: {filename} -> {os.path.basename(new_path)}")

        if rename_dirs:
            for i, dirname in enumerate(dirs):
                new_name = remove_brackets(dirname)
                if new_name == dirname:
                    continue
                new_path = os.path.join(current_root, new_name)
                if os.path.exists(new_path):  # 目录无法合并，跳过
                    print(f"  跳过（目标已存在）: {dirname}")
                    continue
                os.rename(os.path.join(current_root, dirname), new_path)
                dirs[i] = new_name  # 同步给 os.walk，保证还能继续下钻
                print(f"  重命名目录: {dirname} -> {new_name}")

        if not recursive:  # 只处理一层
            break


def move_files_from_subfolders(root_dir, recursive=True, remove_brackets_in_name=True,
                               delete_empty_dirs=True):
    """
    一次性完成：递归把各级子文件夹中的文件移到根目录（顺带去掉文件名方括号）、
    删除空子文件夹，并对根目录原有文件也做一次去方括号重命名。
    """
    if not (root_dir := _resolve_dir(root_dir)):
        return

    print(f"开始处理目录: {root_dir}")

    # 1. 收集子目录中的文件（先收集再移动，避免边遍历边改动目录）
    pending = []
    for current_root, dirs, files in os.walk(root_dir):
        if current_root == root_dir:
            if not recursive:  # 只处理一层子目录
                dirs[:] = []
            continue
        pending.extend(os.path.join(current_root, f) for f in files)

    # 2. 移动到根目录，并去掉文件名中的方括号
    for src_path in pending:
        filename = os.path.basename(src_path)
        dest_name = remove_brackets(filename) if remove_brackets_in_name else filename
        dest_path = _unique_path(root_dir, dest_name)  # 目标已存在时自动加后缀
        shutil.move(src_path, dest_path)
        print(f"  移动文件: {filename} -> {os.path.basename(dest_path)}")

    # 3. 自底向上删除空子文件夹
    if delete_empty_dirs:
        for current_root, dirs, files in os.walk(root_dir, topdown=False):
            if current_root == root_dir:
                continue
            if os.listdir(current_root):  # 如果非空
                print(f"警告：子文件夹非空，未删除 - {current_root}")
                continue
            os.rmdir(current_root)
            print(f"删除空子文件夹: {current_root}")

    # 4. 根目录原有的文件也去掉方括号
    if remove_brackets_in_name:
        _rename_in_dir(root_dir)

    print("处理完成。")


def rename_files_remove_brackets(root_dir, recursive=False, rename_dirs=False):
    """只改名不移动：去掉目录下文件名（rename_dirs 时含目录名）中的方括号。"""
    if not (root_dir := _resolve_dir(root_dir)):
        return

    print(f"开始去除方括号: {root_dir} (recursive={recursive})")

    _rename_in_dir(root_dir, recursive=recursive, rename_dirs=rename_dirs)

    print("处理完成。")


if __name__ == "__main__":
    args = sys.argv[1:]
    flags = {a for a in args if a.startswith("-")}
    target_dir = next((a for a in args if not a.startswith("-") and a not in ("move", "rename")),
                      os.path.dirname(os.path.abspath(__file__)))

    if "rename" in args:
        rename_files_remove_brackets(target_dir,
                                     recursive=bool({"-r", "--recursive"} & flags),
                                     rename_dirs=bool({"-d", "--dirs"} & flags))
    else:
        move_files_from_subfolders(target_dir, recursive="--no-recursive" not in flags)
