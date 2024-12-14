import torch

from lab.tensor_board import to_tensorboard

class lab_TensorBoard():
    def __init__(self):
        self.tensorboard = {"image": [], "graph": [],"scalar":[]}

    def to_tensorboard(self):
        return self.tensorboard

    def _append_model(self,model,input_size):
        """
        :param model: Model
        :param input_size: torch.Size

        :return:
        """
        data = torch.ones([1] + [*input_size])
        self.tensorboard["graph"].append((model,data))

    def _append_image(self,name,img_grid):
        """

        :param name: str
        :param img_grid: tensor[RGB[3],X,Y]
        :return:
        """
        self.tensorboard["image"].append((name, img_grid))

    def _append_graph(self,name,graph):
        self.tensorboard["scalar"].append((name, graph))


class LabObject(lab_TensorBoard):
    def __init__(self, name = "no name has given",information = None):
        self.name = name
        super(LabObject, self).__init__()
        self.information = information
        self.get_information()
        self.set_information()



    def get_information(self):
        raise NotImplementedError( f"{self.name} not implemented get_information")

    def set_information(self):
        raise NotImplementedError(f"{self.name} not implemented set_information")




class Information():
    def __init__(self):
        ####################### data
        self.data = None

        self.classes = None
        self.data_type = None
        self.labels_type = None

        self.train_length = None
        self.train_data_shape = None
        self.train_labels_shape = None
        self.test_length = None

        self.for_loss_target_type = None

        ####################### loss

        self.loss_fn = None
        self.loss_target_type = None

        ####################### layers
        self.layers = None

        ######################## classAdapter
        self.classAdapter = None


        ########################  model

        self.model = None

        ######################### evaluate

        self.evaluate = None


    def to_tensorboard(self):
         to_tensorboard([self.data,self.model,self.model.evaluate])




