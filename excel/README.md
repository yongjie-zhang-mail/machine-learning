# Excel to JSON 转换器

这是一个用于将 Excel 文件转换为 JSON 格式的 Python 程序。

## 功能特性

- 支持批量转换文件夹下的所有 Excel 文件
- 支持多种 Excel 格式 (`.xlsx`, `.xls`, `.xlsm`)
- 每个工作表生成独立的 JSON 文件
- 自动处理中文文件名和工作表名
- 保留原始数据结构和列名
- 提供详细的转换报告

## 安装依赖

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install pandas openpyxl
```

## 使用方法

### 1. 命令行使用

```bash
# 转换当前目录下的所有 Excel 文件
python excel_to_json.py

# 指定输入文件夹
python excel_to_json.py -i /path/to/excel/folder

# 指定输入和输出文件夹
python excel_to_json.py -i /path/to/excel/folder -o /path/to/output/folder

# 指定编码格式
python excel_to_json.py -e gbk
```

### 2. 作为模块使用

```python
from excel_to_json import ExcelToJsonConverter

# 创建转换器实例
converter = ExcelToJsonConverter('/path/to/excel/folder')

# 转换所有 Excel 文件
result = converter.convert_all_excel_files()

# 转换单个文件
excel_files = converter.get_excel_files()
if excel_files:
    result = converter.excel_to_json(excel_files[0])
```

### 3. 运行示例

```bash
python example_usage.py
```

## 输出格式

每个 Excel 工作表会生成一个 JSON 文件，格式如下：

```json
{
  "sheet_name": "工作表名称",
  "total_rows": 行数,
  "total_columns": 列数,
  "columns": ["列1", "列2", "列3"],
  "data": [
    {"列1": "值1", "列2": "值2", "列3": "值3"},
    {"列1": "值4", "列2": "值5", "列3": "值6"}
  ]
}
```

## 文件命名规则

生成的 JSON 文件命名格式为：`{Excel文件名}_{工作表名}.json`

例如：
- `data.xlsx` 的 `Sheet1` 工作表 → `data_Sheet1.json`
- `体能概况.xlsx` 的 `数据` 工作表 → `体能概况_数据.json`

## 注意事项

1. 程序会自动处理 Excel 中的空值（NaN），将其转换为空字符串
2. 工作表名中的特殊字符会被清理，确保文件名的有效性
3. 支持中文文件名和内容
4. 建议使用 UTF-8 编码以确保中文字符正确显示

## 错误处理

程序具有完善的错误处理机制：
- 文件不存在或无法读取时会显示详细错误信息
- 损坏的 Excel 文件会跳过，不影响其他文件的转换
- 转换完成后会显示成功和失败的统计信息
