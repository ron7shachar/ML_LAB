from lab.tensor_board import to_tensorboard

class LabObject():
    def __init__(self, name = "no name has given",information = None):
        self.name = name
        self.tensorboard = {}
        self.information = information
        self.get_information()
        self.set_information()

    def to_tensorboard(self):
        return self.tensorboard

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


    def to_tensorboard(self):
         to_tensorboard([self.data,self.model])




