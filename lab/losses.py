from sympy.physics.units.definitions.dimension_definitions import information
from torch import nn
from lab.lab_ import LabObject


class LabLosse(LabObject):
    def __init__(self,name ="Losse",*attributes ):
        """self.target_type = ""  # "classes" , "index" , "prob" , "" """
        super().__init__(name,*attributes)


    def get_information(self):
        pass
    def set_information(self):
        information = self.information
        information.loss_fn = self
        information.loss_target_type = self.target_type



class MSELoss(nn.MSELoss,LabLosse):
    def __init__(self,*attributes ):
        self.target_type = ""  # "classes" , "index" , "prob" , ""
        LabLosse.__init__(self, "MSELoss",*attributes)
        nn.MSELoss.__init__(self)






loss_table = {
    "MSELoss":MSELoss

}

def get_loss(loss_name,*attributes):
    """

    :param loss_name:
    :param attributes:

    MSELoss

    :return:
    """
    return loss_table[loss_name](*attributes)