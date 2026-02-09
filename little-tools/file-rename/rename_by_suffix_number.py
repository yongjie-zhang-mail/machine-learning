import os
import re
import argparse


def extract_suffix_number(filename):
    """
    从文件名中提取末尾的数字后缀。
    例如: "PostTraining_LLMs_M1_10.png" -> 10
    """
    name_without_ext = os.path.splitext(filename)[0]
    match = re.search(r'(\d+)$', name_without_ext)
    if match:
        return int(match.group(1))
    return None


def rename_files_by_suffix_number(directory, start_number, dry_run=True):
    """
    根据文件名后缀数字升序排序，然后从 start_number 开始依次重命名。

    Args:
        directory (str): 目标文件夹路径
        start_number (int): 起始数字
        dry_run (bool): 为 True 时仅预览，不实际重命名
    """
    # 获取目录下所有文件（排除目录和本脚本自身）
    script_name = os.path.basename(__file__)
    files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and f != script_name
    ]

    # 筛选出文件名末尾带数字的文件
    files_with_number = []
    for f in files:
        num = extract_suffix_number(f)
        if num is not None:
            files_with_number.append((f, num))

    if not files_with_number:
        print("未找到文件名末尾带数字的文件。")
        return

    # 按后缀数字升序排序
    files_with_number.sort(key=lambda x: x[1])

    print(f"找到 {len(files_with_number)} 个文件，将从 {start_number} 开始重命名：\n")

    # 构建重命名计划
    rename_plan = []
    for i, (filename, old_num) in enumerate(files_with_number):
        new_num = start_number + i
        name_without_ext = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]

        # 替换末尾数字为新数字
        new_name = re.sub(r'\d+$', str(new_num), name_without_ext) + ext
        rename_plan.append((filename, new_name))
        print(f"  {filename}  ->  {new_name}")

    if dry_run:
        print("\n[预览模式] 未执行实际重命名，添加 --run 参数以执行。")
        return

    # 为避免重命名冲突，先全部改为临时名称，再改为目标名称
    temp_names = []
    for old_name, new_name in rename_plan:
        temp_name = f"__temp_rename__{old_name}"
        os.rename(
            os.path.join(directory, old_name),
            os.path.join(directory, temp_name),
        )
        temp_names.append((temp_name, new_name))

    for temp_name, new_name in temp_names:
        os.rename(
            os.path.join(directory, temp_name),
            os.path.join(directory, new_name),
        )

    print(f"\n已完成重命名，共处理 {len(rename_plan)} 个文件。")


def main():
    parser = argparse.ArgumentParser(
        description="根据文件名后缀数字升序排序，从指定起始数字开始依次重命名。"
    )
    parser.add_argument(
        "start_number",
        type=int,
        help="重命名的起始数字",
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        default=".",
        help="目标文件夹路径（默认为当前目录）",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="实际执行重命名（默认为预览模式）",
    )

    args = parser.parse_args()
    directory = os.path.abspath(args.directory)

    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在。")
        return

    rename_files_by_suffix_number(directory, args.start_number, dry_run=not args.run)

# 预览模式: python rename_by_suffix_number.py 100
# 实际执行: python rename_by_suffix_number.py 100 --run
if __name__ == "__main__":
    main()
