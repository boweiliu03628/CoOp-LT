import os

# 数据集文件夹路径
dataset_path = "/home/bowei/Desktop/PyProjs/myb1/caltech-101/train"

# 结果保存路径
output_txt_path = "processed_caltech101_classnames.txt"

def sort_and_save_folders(dataset_path, output_txt_path):
    # 获取目录中的文件夹名称（仅限文件夹）
    class_names = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    # 按字母顺序排序
    sorted_class_names = sorted(class_names)
    
    # 写入txt文件
    with open(output_txt_path, "w") as f:
        for class_name in sorted_class_names:
            f.write(class_name + "\n")
    
    print(f"排序结果已保存到: {output_txt_path}")

# 调用函数
sort_and_save_folders(dataset_path, output_txt_path)
