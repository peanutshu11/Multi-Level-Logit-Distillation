from torchvision import datasets
from .cifar100 import get_data_folder, get_cifar100_train_transform, get_cifar100_test_transform
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import torch


class KCDCIFAR100(datasets.CIFAR100):
    def __init__(self, root, train = True, download=False, transform = None, target_transform = None,
    threshold = 0.5, savedir = None):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.remain = np.array([i for i in range(len(self.data))])
        self.threshold = threshold
        self.entropy_record = None
        self.mix_up_data = torch.tensor(self.data, requires_grad=False)


    def __getitem__(self, index):
        if self.train:
            img, target = self.mix_up_data[self.remain[index]].numpy(), self.targets[self.remain[index]]
        else:
            img, target = self.test_data[self.remain[index]], self.test_labels[self.remain[index]]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.remain[index]
    
    def __len__(self):
        return len(self.remain)
    
    @torch.no_grad()
    def mixup(self):
        if len(self.mix_remain) == 0 or len(self.lost) == 0:
            return
        remain_idx = torch.tensor(self.mix_remain.copy())
        lost_idx   = torch.tensor(self.lost.copy())
        remain_data = self.mix_up_data[remain_idx]
        lost_data   = self.mix_up_data[lost_idx]
        
        mix_len = len(remain_idx)
        idx = torch.arange(mix_len, device=remain_data.device)
        
        remain_rate = (1 - 0.7) * ((mix_len - idx) / mix_len) + 0.7
        lost_rate = 1.0 - remain_rate

        # broadcast
        view_shape = [mix_len] + [1] * (remain_data.dim() - 1)
        remain_rate = remain_rate.view(view_shape)
        lost_rate = lost_rate.view(view_shape)
        mixup = remain_data * remain_rate + lost_data * lost_rate

        self.mix_up_data[remain_idx] = mixup.to(torch.uint8)


    def init_len(self):
        return len(self.data)

    def update_statistics(self, entropy):
        if self.entropy_record is None:
            self.entropy_record = entropy.reshape((len(entropy),1))
        else:
            self.entropy_record = np.concatenate((self.entropy_record, entropy.reshape((len(entropy),1))),axis=1)


    def update_remain(self, remain):
        self.remain = remain
    
    def update_dataset(self):
        if len(self.remain) - int(len(self.data) * (1 - self.threshold) / 6) < int(self.threshold * len(self.data)):
            remain_num = int(self.threshold * len(self.data))
        else:
            remain_num = len(self.remain) - int(len(self.data) * (1 - self.threshold) / 6)

        _entropy = np.clip(self.entropy_record,0.0,a_max = self.entropy_record.max())
        _entropy = np.sum(_entropy, axis=1)
        _sum = np.where(self.entropy_record > 0.0, 1.0, 0.0)
        _sum = np.sum(_sum, axis=1)
        ET = np.divide(_sum, 40.0)
        ET = np.power(ET, 0.03)
        _entropy = np.divide(_entropy, _sum)
        scores = ET * _entropy
        indice = scores.argsort()[::-1]
        pre_len = len(self.remain)
        self.remain = indice[:remain_num]
        self.lost = indice[remain_num:pre_len]
        self.mix_remain = self.remain[-len(self.lost):]
        self.mixup()
    

def get_kcd_dataloader(batch_size, val_batch_size, num_workers):
    data_folder = get_data_folder()
    train_transform = get_cifar100_train_transform()
    test_transform = get_cifar100_test_transform()
    train_set = KCDCIFAR100(
        root=data_folder, download=True, train=True, transform=train_transform
    )
    num_data = len(train_set)
    test_set = datasets.CIFAR100(
        root=data_folder, download=True, train=False, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=1,
    )
    return train_loader, test_loader, num_data, train_set
