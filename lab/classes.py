import torch
from torch.utils.hipify.hipify_python import value

from lab.lab_ import LabObject
from losses import LabLosse
from torch import nn

# class LabClass(LabObject):
#     def __init__(self,name = "Class"):
#         super().__init__(name)


class ClassAdapter(LabObject):
    def __init__(self,information):
        super().__init__('classAdapter',information)

        self.classtoind = {c: i for i, c in enumerate(self.classes)}
        self.indtoclass = {i: c for i, c in enumerate(self.classes)}
        self.length = len(self.classes)

        self.value = {
            ("classes","classes") :lambda x:x.float(),
            ("classes", "index"): lambda x:self.classtoind[x].float(),
            ("classes", "prob"): lambda x: nn.functional.one_hot(self.classtoind[x],self.length).float(),
            ("classes", ""): lambda x: nn.functional.one_hot(self.classtoind[x],self.length).float(),

            ("index","index") :lambda x:x.float(),
            ("index", "classes"): lambda x: self.indtoclass[x].float(),
            ("index", "prob"): lambda x: nn.functional.one_hot(x,self.length).float(),
            ("index", ""): lambda x: nn.functional.one_hot(x,self.length).float(),

            ("prob","prob") :lambda x:x.float(),
            ("prob", "classes"): lambda x: self.indtoclass(torch.argmax(x)),
            ("prob", "index"): lambda x: torch.argmax(x),
            ("prob", ""): lambda x: x.float(),

            ("","") :lambda x: x.float(),
            ("", "classes"): lambda x: self.indtoclass(torch.argmax(x)),
            ("", "index"): lambda x: torch.argmax(x),
            ("", "prob"): lambda x: x/torch.sum(x),

        }

    def get_information(self):
        information = self.information
        self.classes = information.classes
        self.key = (information.for_loss_target_type, information.loss_target_type)


    def set_information(self):
        information = self.information
        information.classAdapter = self




    def __call__(self, prediction , target):
        return prediction , self.value[self.key](target)




