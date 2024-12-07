from torch.utils.data import Dataset
import pickle
import torch

class CustomDataset(Dataset):
    def __init__(self, pkl_path):
        self.classnames = None

        with open(pkl_path, 'rb') as f:
            self.data_dict = pickle.load(f)  # 包含 {label: (n, 512) numpy arrays} 的字典

        # 将数据展平成 [(tensor, label)] 的形式
        self.samples = []
        for label, embeddings in self.data_dict.items():
            tensors = torch.tensor(embeddings)
            self.samples.extend([(tensors[i], label) for i in range(tensors.shape[0])])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        embedding, label = self.samples[idx]
        return embedding, label