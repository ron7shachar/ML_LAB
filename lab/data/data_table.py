import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset
from lab.lab_ import LabObject

from torch.utils.tensorboard import SummaryWriter

class Lab_dataset(LabObject):
    def __init__(self,name,*inputs):
        """self.data_train = VisionDataset #train=True
        self.data_test = VisionDataset ##train=False"""
        super(Lab_dataset, self).__init__(name,*inputs)
        self.information.data = self

    def get_information(self):
        pass
    def set_information(self):
        """
        self.classes = None
        self.data_type = None
        self.labels_type = None

        self.train_length = None
        self.train_data_shape = None
        self.train_labels_shape = None
        self.test_length = None
        self.for_loss_target_type    # "classes" , "index" , "prob" , "" """
        raise NotImplementedError("not implemented set_information")


class Number_classification(Lab_dataset):

    def __init__(self,*inputs):
        self.data_train = datasets.MNIST(root='./lab/data/data_storage', train=True, download=True)
        self.data_test = datasets.MNIST(root='./lab/data/data_storage', train=False, download=True)
        super().__init__("number classification",*inputs)

        self.set_properties()

    def set_information(self):
        pass
    def set_properties(self):
        information = self.information

        information.classes = self.data_train.classes
        information.data_type = self.data_train_data.dtype
        information.labels_type = self.data_train_data.dtype

        information.train_length = self.data_train_data.shape[0]
        information.train_data_shape = self.data_train_data[0].shape
        information.train_labels_shape = self.data_train_data[0].shape
        information.test_length = self.data_test_data.shape[0]




    def set_properties(self):
        self.data_train_data = self.data_train.data
        self.data_train_labels = self.data_train.targets

        self.data_test_data = self.data_test.data
        self.data_test_labels = self.data_test.targets

        self.properties = {
            "classes": self.data_train.classes,
            "data_type": self.data_train_data.dtype,
            "labels_type": self.data_train_data.dtype,

            "train_length": self.data_train_data.shape[0],
            "train_data_shape": self.data_train_data[0].shape,
            "train_labels_shape": self.data_train_data[0].shape,

            "test_length": self.data_test_data.shape[0],
            "test_data_shape": self.data_test_data[0].shape,
            "test_labels_shape": self.data_test_data[0].shape,

        }

    def set_information(self):
        information = self.information

        information.classes = self.data_train.classes
        information.data_type = self.data_train.data.dtype
        information.labels_type = self.data_train.targets.dtype

        information.train_length = self.data_train.data.shape[0]
        information.train_data_shape = self.data_train.data[0].shape
        information.train_labels_shape = self.data_train.targets.unsqueeze(1)[0].shape
        information.test_length = self.data_test.data.shape[0]

        img_grid = torchvision.utils.make_grid(self.data_train.data[:64].unsqueeze(1))
        self._append_image('mnist_images',img_grid)

        information.for_loss_target_type = "index"
data_table = {"number_classification": Number_classification}

def get_data(name,*inputs) -> Lab_dataset:
    """
    "number_classification"
    :param name:
    :param inputs:



    :return:
    """
    return data_table[name](*inputs)