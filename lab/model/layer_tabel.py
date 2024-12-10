import torch
from torch import nn, Tensor
from lab.lab_ import LabObject

class Lab_layer(LabObject):
    def __init__(self,name):
        super(Lab_layer,self).__init__(name)

    def get_information(self):
        pass
    def set_information(self):
        pass


################################### static layer
class Reshape(nn.Module,Lab_layer):
    def __init__(self,shape_in  , shape_out):
        nn.Module.__init__(self)
        LabObject.__init__(self,f"reshape {shape_in} -> {shape_out}",None)
        self.shape_in = shape_in
        self.shape_out = shape_out

    def forward(self,x:Tensor):
        return x.reshape([-1]+self.shape_out).float()

class Sigmoid(nn.Module, Lab_layer):
    def __init__(self,in_features, out_features):
        if in_features != out_features:
            raise KeyError("sigmoid do not change the dimension")
        nn.Module.__init__(self)
        LabObject.__init__(self, f"sigmoid")
    def forward(self,x:Tensor):
        return torch.sigmoid(x)


################################### optimiz layer
class Linear(nn.Linear, Lab_layer):  # Ensure Lab_layer is after nn.Linear
    def __init__(self, in_features, out_features, bias=True):
        nn.Linear.__init__(self, in_features[0], out_features[0], bias)  # Call nn.Linear's __init__
        Lab_layer.__init__(self, 'Linear')  # Call Lab_layer's __init__ with the name 'Linear'


layer_table = {
## optimiz layer
    "Linear":Linear,
### static layer
    "Reshape":Reshape,
    "Sigmoid":Sigmoid
}
def get_layer(name,input_size,output_size,*layer_attributes):
    return layer_table[name](input_size,output_size,*layer_attributes)

def get_layers(layers,information):
        """

        :param layers:
        :param information:

        ## optimiz layer
        "Linear":Linear,
        ### static layer
        "Reshape":Reshape,
        "Sigmoid":Sigmoid

        :return:
        """
        layers_ = [information.train_data_shape] + layers + [[len(information.classes)]]
        less = layers_[0]
        is_name = False
        new_layers = layers_[:1]
        for i in layers_[1:]:
            if (isinstance(i, str) or isinstance(i, tuple)):
                if is_name:
                    new_layers.append(less)
                else:
                    is_name = True
            else:
                less = i
                is_name = False
            new_layers.append(i)
        print(new_layers)

        input_size = None
        output_size = None
        layers = []
        next_layer = None
        for i in new_layers:
            if (isinstance(i, str) or isinstance(i, tuple)):
                next_layer = i
            else:
                input_size = output_size
                output_size = i
                if not next_layer is None:
                    if isinstance(next_layer, tuple):
                        j, a = next_layer
                        layers.append(get_layer(j, input_size, output_size, a))
                    else:
                        layers.append(get_layer(next_layer, input_size, output_size))
        information.layers = layers
        return layers