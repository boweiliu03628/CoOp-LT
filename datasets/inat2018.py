import json
from .custom_dataset import CustomDataset


class iNaturalist2018(CustomDataset):
    def __init__(self, pkl_path):
        # 通过调用父类构造函数初始化数据
        super().__init__(pkl_path)
        file_path = 'iNaturalist2018/categories.json'
        with open(file_path, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        # Create a dictionary to map id to name
        id_to_name = {item['id']: item['name'] for item in categories}
        # Generate a sorted list of names based on id from 0 to 8141
        name_list = [id_to_name[i] for i in range(8142) if i in id_to_name]
        self.classnames = name_list

