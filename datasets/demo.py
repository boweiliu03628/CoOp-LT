txt_file_path = "caltech101/processed_caltech101_classnames.txt"
class_names = []
with open(txt_file_path, "r") as file:
    for line in file:
        # 去掉行尾的换行符和多余空格
        class_name = line.strip()
        if class_name:  # 确保非空行才加入列表
            class_names.append(class_name)

# 打印读取的列表
print("类名列表:", class_names)