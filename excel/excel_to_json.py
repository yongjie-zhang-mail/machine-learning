#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Excel to JSON 转换器
将指定文件夹下的所有 Excel 文件转换为 JSON 格式
"""

import pandas as pd
import json
import os
import sys
from pathlib import Path


class ExcelToJsonConverter:
    """Excel 到 JSON 转换器类"""
    
    def __init__(self, excel_folder_path=None):
        """
        初始化转换器
        
        Args:
            excel_folder_path (str): Excel 文件所在文件夹路径，默认为当前目录
        """
        if excel_folder_path is None:
            self.excel_folder_path = Path.cwd()
        else:
            self.excel_folder_path = Path(excel_folder_path)
        
        if not self.excel_folder_path.exists():
            raise FileNotFoundError(f"文件夹不存在: {self.excel_folder_path}")
    
    def get_excel_files(self):
        """
        获取文件夹下所有 Excel 文件
        
        Returns:
            list: Excel 文件路径列表
        """
        excel_extensions = ['.xlsx', '.xls', '.xlsm']
        excel_files = []
        
        for file_path in self.excel_folder_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in excel_extensions:
                excel_files.append(file_path)
        
        return excel_files
    
    def excel_to_json(self, excel_file_path, output_folder=None, encoding='utf-8'):
        """
        将单个 Excel 文件转换为 JSON
        
        Args:
            excel_file_path (Path): Excel 文件路径
            output_folder (str): 输出文件夹，默认为 Excel 文件所在文件夹
            encoding (str): JSON 文件编码格式，默认为 utf-8
        
        Returns:
            dict: 转换结果信息
        """
        try:
            # 设置输出文件夹
            if output_folder is None:
                output_folder = excel_file_path.parent
            else:
                output_folder = Path(output_folder)
                output_folder.mkdir(exist_ok=True)
            
            # 读取 Excel 文件的所有工作表
            excel_data = pd.read_excel(excel_file_path, sheet_name=None, engine='openpyxl')
            
            # 准备转换结果
            result = {
                'file_name': excel_file_path.name,
                'total_sheets': len(excel_data),
                'sheets_converted': [],
                'output_files': []
            }
            
            # 为每个工作表创建单独的 JSON 文件
            for sheet_name, df in excel_data.items():
                # 处理 NaN 值
                df_cleaned = df.fillna("")
                
                # 转换为字典格式
                json_data = {
                    'sheet_name': sheet_name,
                    'total_rows': len(df_cleaned),
                    'total_columns': len(df_cleaned.columns),
                    'columns': df_cleaned.columns.tolist(),
                    'data': df_cleaned.to_dict('records')
                }
                
                # 生成输出文件名
                base_name = excel_file_path.stem
                safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '-', '_')).strip()
                if safe_sheet_name:
                    json_filename = f"{base_name}_{safe_sheet_name}.json"
                else:
                    json_filename = f"{base_name}_sheet_{len(result['sheets_converted'])+1}.json"
                
                json_file_path = output_folder / json_filename
                
                # 写入 JSON 文件
                with open(json_file_path, 'w', encoding=encoding) as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                
                result['sheets_converted'].append(sheet_name)
                result['output_files'].append(str(json_file_path))
                
                print(f"✓ 已转换工作表 '{sheet_name}' -> {json_filename}")
            
            result['status'] = 'success'
            return result
            
        except Exception as e:
            return {
                'file_name': excel_file_path.name,
                'status': 'error',
                'error_message': str(e)
            }
    
    def convert_all_excel_files(self, output_folder=None, encoding='utf-8'):
        """
        转换文件夹下所有 Excel 文件
        
        Args:
            output_folder (str): 输出文件夹，默认为 Excel 文件所在文件夹
            encoding (str): JSON 文件编码格式，默认为 utf-8
        
        Returns:
            dict: 总体转换结果
        """
        excel_files = self.get_excel_files()
        
        if not excel_files:
            print("❌ 在指定文件夹中未找到任何 Excel 文件")
            return {'total_files': 0, 'successful': 0, 'failed': 0, 'results': []}
        
        print(f"📁 在文件夹 {self.excel_folder_path} 中找到 {len(excel_files)} 个 Excel 文件")
        print("-" * 60)
        
        results = []
        successful = 0
        failed = 0
        
        for excel_file in excel_files:
            print(f"📊 正在处理: {excel_file.name}")
            result = self.excel_to_json(excel_file, output_folder, encoding)
            results.append(result)
            
            if result['status'] == 'success':
                successful += 1
                print(f"   ✅ 成功转换 {result['total_sheets']} 个工作表")
            else:
                failed += 1
                print(f"   ❌ 转换失败: {result['error_message']}")
            
            print()
        
        # 输出总结
        print("=" * 60)
        print(f"📈 转换完成!")
        print(f"   总文件数: {len(excel_files)}")
        print(f"   成功: {successful}")
        print(f"   失败: {failed}")
        
        return {
            'total_files': len(excel_files),
            'successful': successful,
            'failed': failed,
            'results': results
        }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='将 Excel 文件转换为 JSON 格式')
    parser.add_argument('-i', '--input', type=str, default='.', 
                       help='Excel 文件所在文件夹路径 (默认: 当前目录)')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='JSON 文件输出文件夹路径 (默认: 与输入文件夹相同)')
    parser.add_argument('-e', '--encoding', type=str, default='utf-8',
                       help='JSON 文件编码格式 (默认: utf-8)')
    
    args = parser.parse_args()
    
    try:
        # 创建转换器
        converter = ExcelToJsonConverter(args.input)
        
        # 执行转换
        result = converter.convert_all_excel_files(args.output, args.encoding)
        
        # 根据结果设置退出码
        if result['failed'] == 0:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
