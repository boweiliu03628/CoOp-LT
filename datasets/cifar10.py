from torchvision.datasets import CIFAR10
from .custom_dataset import CustomDataset


class CIFAR10_LT(CustomDataset):
    def __init__(self, pkl_path):
        # 通过调用父类构造函数初始化数据
        super().__init__(pkl_path)
        data_dir = "/home/bowei/Desktop/PyProjs/myb1/cifar10_dataset"
        dataset = CIFAR10(root=data_dir, download=False)
        self.classnames = dataset.classes
        # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
