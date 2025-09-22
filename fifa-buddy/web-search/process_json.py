import json
import sys
import os

def main(json_data) -> dict:
    """
    从传入的已解析 JSON 对象(字典或列表)中提取 'webpage' 键的值，
    并将其作为字符串返回，该字符串已正确转义以用作另一个 JSON 的值。

    Args:
        json_data (dict | list): 已解析的 JSON 数据（非字符串）。

    Returns:
        dict: 包含 "result" 键的字典，其值为 'webpage' 内容的 JSON 编码字符串。
    """
    try:
        if not isinstance(json_data, (dict, list)):
            return {"error": "Input must be a dict or list (parsed JSON object)."}

        data = json_data
        webpage_content = ""

        if isinstance(data, dict):
            webpage_content = data.get('webpage', '')
        else:  # list
            for item in data:
                if isinstance(item, dict) and 'webpage' in item:
                    webpage_content = item.get('webpage', '')
                    break

        escaped_string = json.dumps(webpage_content)
        return {"result": escaped_string}
    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(script_dir, 'original-json.json')

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_content_raw = f.read()
        json_content = json.loads(json_content_raw)
    except FileNotFoundError:
        print(f"Error: '{json_file_path}' not found. The script will now exit.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {json_file_path}")
        sys.exit(1)

    result = main(json_content)

    if "error" in result:
        print(f"An error occurred: {result['error']}")
    else:
        print("--- Escaped string for JSON value ---")
        print(result['result'])

        new_json_obj = {
            "source": "from json object",
            "webpage_as_string": result['result']
        }

        print("\n--- Example of a new JSON object containing the escaped string ---")
        print(json.dumps(new_json_obj, indent=2))

