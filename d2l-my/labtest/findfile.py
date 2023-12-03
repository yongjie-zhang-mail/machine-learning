# Import os and json modules
import os
import json

# 使用 LLM 大预言模型 进行编程辅助 试验
# 先使用 New Bing 进行开放性的输出，再使用 CodeWhisperer 进行裁剪，最后人工调整
# 在调试过程中，可以将保存信息再抛给 New Bing


# python编程需求：
# 对一个windows目录下的所有文件进行一些统计工作，输入是一个windows目录，输出是一个在此目录下的output.txt文档。
# 逻辑和步骤：
# 1.遍历这个windows目录，对于每个找到的文件夹继续找到它的下一级子文件夹，直到在目录下找到一个名叫entry.json的文件。
# 2.读取这个entry.json文件，读取"title"属性的值，若含有"韩国AFreecaTV-"，则读取出来后面的两个字符保存为"category"变量，同时读取该entry.json文件中的"avid"属性的值，保存为"id"变量。
# 3.将遍历每个文件夹得到的"category"变量和"id"变量作为一行保存进output.txt文档。
#
# python programming requirement: Perform some statistical work on all files in a windows directory, input is a windows directory, output is an output.txt document in this directory. Logic and steps:
#
# 1.Traverse this windows directory, for each folder found, continue to find its next level subfolder until a file named entry.json is found in the directory.
# 2.Read this entry.json file, read the value of the "title" attribute, if it contains "韩国AFreecaTV-", then read out the next two characters after it and save it as the "category" variable, and read the value of the "avid" attribute in the entry.json file and save it as the "id" variable.
# 3.Save the "category" variable and "id" variable obtained by traversing each folder as one line into the output.txt document.


# define a function to find the entry.json file
def find_entry_json():
    # Define the input directory
    input_dir = "D:\\B站视频\\临时"
    

    # Define the output file
    output_file = "D:\\B站视频\\临时\\output.txt"

    # Open the output file in write mode
    with open(output_file, "w", encoding="utf-8") as f:

        # Traverse the input directory using os.walk
        for root, dirs, files in os.walk(input_dir):

            # Check if entry.json is in the files
            if "entry.json" in files:

                # Construct the full path of entry.json
                entry_path = os.path.join(root, "entry.json")
                
                # Open the entry.json file in read mode
                with open(entry_path, "r", encoding="utf-8") as e:

                    # Load the json data as a dictionary
                    data = json.load(e)

                    # Get the value of the title attribute
                    title = data["title"]

                    # Check if it contains "韩国AFreecaTV-"
                    if "韩国AFreecaTV-" in title:

                        # Get the next two characters after it as category
                        category = title[title.index("韩国AFreecaTV-") + len("韩国AFreecaTV-"):title.index("韩国AFreecaTV-") + len("韩国AFreecaTV-") + 2]

                        # Get the value of the avid attribute as id
                        id = data["avid"]

                        # Write the category and id as one line to the output file, separated by a comma
                        f.write(category + "," + str(id) + "\n")

# 上面output.txt文件中每一行的文本都是逗号分隔的，对逗号使用split方法将每行处理成一个数组。
# 以每行对应的数组的一个元素作为Key对这些行进行排序。
# 将排序好的内容写入和output.txt同目录下的output-ordered.txt。
#
# 1.Each line of text in the output.txt file above is comma-separated, use the split method to process each line into an array.
# 2.Sort these lines by the first element of the array corresponding to each line as the Key.
# 3.Write the sorted content to output-ordered.txt in the same directory as output.txt.


# define a function to order the output file
def order_output_file():
    # Define the input file
    input_file = "D:\\B站视频\\临时\\output.txt"

    # Define the output file
    output_file = "D:\\B站视频\\临时\\output_ordered.txt"

    # Create an empty list to store the lines
    lines = []

    # Open the input file in read mode
    with open(input_file, "r", encoding="utf-8") as f:

        # Loop through each line in the file
        for line in f:

            # Strip the newline character from the line
            line = line.strip()

            # Split the line by comma and append it to the list as a sub-list
            lines.append(line.split(","))

    # Sort the list by the first element of each sub-list using the sorted function
    lines = sorted(lines, key=lambda x: x[0])

    # Open the output file in write mode
    with open(output_file, "w", encoding="utf-8") as f:

        # Loop through each sub-list in the sorted list
        for line in lines:

            # Join the sub-list elements by comma and write it to the output file with a newline character
            f.write(",".join(line) + "\n")


def find_entry_json_with_specific_title():

    # Define the input directory
    input_dir = "D:\\B站视频\\临时"

    # Define the output file
    output_file = "D:\\B站视频\\临时\\specific_title_output.txt"

    # Open the output file in write mode
    with open(output_file, "w", encoding="utf-8") as f:

        # Traverse the input directory using os.walk
        for root, dirs, files in os.walk(input_dir):

            # Check if entry.json is in the files
            if "entry.json" in files:

                # Construct the full path of entry.json
                entry_path = os.path.join(root, "entry.json")

                # Open the entry.json file in read mode
                with open(entry_path, "r", encoding="utf-8") as e:

                    # Load the json data as a dictionary
                    data = json.load(e)

                    # Get the value of the title attribute
                    specific_attr = data["title"]

                    # Check if it contains "xxx"
                    if "塞拉" in specific_attr:

                        # Get the value of the avid attribute as id
                        id = data["avid"]

                        # Write to the output file
                        f.write(str(id) + "\n")


def find_entry_json_with_specific_owner_name():

    # Define the input directory
    input_dir = "D:\\B站视频\\临时"

    # Define the output file
    output_file = "D:\\B站视频\\临时\\specific_owner_output.txt"

    # Open the output file in write mode
    with open(output_file, "w", encoding="utf-8") as f:

        # Traverse the input directory using os.walk
        for root, dirs, files in os.walk(input_dir):

            # Check if entry.json is in the files
            if "entry.json" in files:

                # Construct the full path of entry.json
                entry_path = os.path.join(root, "entry.json")

                # Open the entry.json file in read mode
                with open(entry_path, "r", encoding="utf-8") as e:

                    # Load the json data as a dictionary
                    data = json.load(e)

                    # Get the value of the owner_name attribute
                    specific_attr = data["owner_name"]

                    # Check if it contains "xxx"
                    if "阿巍说楼市" in specific_attr:

                        # Get the value of the avid attribute as id
                        id = data["avid"]

                        # Write to the output file
                        f.write(str(id) + "\n")

    # 打印日志
    print("find_entry_json_with_specific_owner_name() is finished. 输出文件：specific_owner_output.txt")




# the main function
if __name__ == "__main__":
    print("Hello, World!")
    # find_entry_json()
    # order_output_file()

    find_entry_json_with_specific_owner_name()

    # find_entry_json_with_specific_title()


    print("Goodbye, World!")





