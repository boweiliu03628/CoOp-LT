import json
from .custom_dataset import CustomDataset


class ImageNet_LT(CustomDataset):
    def __init__(self, pkl_path):
        # 通过调用父类构造函数初始化数据
        super().__init__(pkl_path)
        json_file_path = "ImageNet_LT/imagenet_class_index.json"
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        # sorted_second_values = [value[1] for key, value in sorted(data.items(), key=lambda x: int(x[0]))]
        sorted_second_values = [value[1] for key, value in sorted(data.items(), key=lambda x: x[0])]
        self.classnames = sorted_second_values
