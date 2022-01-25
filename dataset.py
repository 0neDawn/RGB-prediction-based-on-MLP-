import torch
import numpy as np
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,data_path):
        self.data_path = data_path
        self.load_data(data_path)
    def load_data(self, data_path):
        data = np.loadtxt(data_path)
        self.size = data.shape[0]
        self.X = data[:, :3]
        self.Y = data[:, 3:]
    def __len__(self):
        return self.size
    def __iter__(self):
        return self
    def __getitem__(self, index):
        return self.X[index, :], self.Y[index]
