import numpy as np
import torch

from torch import nn
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from lab.lab_ import LabObject


class Dataseting(Dataset):
    def __init__(self, data):
        self.data = data.data
        self.targets = data.targets
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

class Model(Module,LabObject):
    def __init__(self,information,name = "Model"):
        super(Model,self).__init__()
        LabObject.__init__(self, name, information)
        # set the layer

        for i,layer in enumerate(self.layers):
            setattr(self, layer.name+f"_{i}", layer)
        self.tensorboard["graph"] = []



    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def fit(self, data, batch_size=1,epoch = 1, lr=0.03,optimizer=torch.optim.Adam):
        optimizer = optimizer(self.parameters(), lr=lr)
        data = Dataseting(data)
        total_loss = 0
        i = 0
        for epoch in range(epoch):
            for batch in DataLoader(data, batch_size=batch_size):
                x, y = batch
                prediction = self.forward(x)
                loss = self.loss_fn(*self.classAdapter(prediction, y))
                total_loss += loss.item()  # Accumulate scalar value of loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                i += 1
                if i % 1000 == 0:
                    print(total_loss / 1000)
                    i = 0
                    total_loss = 0

    def predict(self,data): # return class
       pass

    def evaluate(self,data):
        pass
    def to_tensorboard(self):
        for param in self.parameters():
            param = param.float().clone()

        self.tensorboard["graph"].append((self,self.data))
        return self.tensorboard
    def get_information(self):
        information = self.information
        self.data = torch.ones([1]+[*information.train_data_shape])
        self.loss_fn =  information.loss_fn
        self.classAdapter = information.classAdapter
        self.layers = information.layers

    def set_information(self):
        information = self.information
        information.model = self


