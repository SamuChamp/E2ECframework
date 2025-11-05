import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


class DataGen:
    def __init__(self, dataset) -> None:
        self.call(dataset)

    def call(self, dataset= 'mnist'):
        """
        load specified dataset.
        """
        if dataset == 'mnist':
            os.makedirs('./data/mnist/', exist_ok=True)
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            self.d_train = datasets.MNIST('./data/mnist/', train= True, download= True, transform= trans_mnist)
            self.d_test = datasets.MNIST('./data/mnist/', train= False, download= True, transform= trans_mnist)
            self.dim = (int(torch.prod(torch.tensor(self.d_train[0][0].shape))), 10)

        else: pass # TODO: add other datasets.

    def batch(self, train= True, batch_size= 20, shuffle= True):
        if train: return DataLoader(self.d_train, batch_size, shuffle)
        else: return DataLoader(self.d_test, batch_size, shuffle=False)
