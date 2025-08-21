#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel to JSON 转换器使用示例
"""

from excel_to_json import ExcelToJsonConverter

def example_usage():
    """使用示例"""
    print("=== Excel to JSON 转换器使用示例 ===\n")
    
    # 方法1: 使用默认设置（当前目录）
    print("1. 转换当前目录下的所有 Excel 文件:")
    converter = ExcelToJsonConverter()
    result = converter.convert_all_excel_files()
    
    print(f"\n转换结果: {result['successful']}/{result['total_files']} 个文件成功转换\n")
    
    # 方法2: 指定特定文件夹
    # converter = ExcelToJsonConverter('/path/to/excel/folder')
    # result = converter.convert_all_excel_files('/path/to/output/folder')
    
    # 方法3: 转换单个文件
    # from pathlib import Path
    # excel_files = converter.get_excel_files()
    # if excel_files:
    #     result = converter.excel_to_json(excel_files[0])
    #     print(f"单个文件转换结果: {result}")

if __name__ == "__main__":
    example_usage()
