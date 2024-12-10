from lab.lab_ import Information
from lab.data.data_table import get_data
from lab.model.layer_tabel import get_layers
from lab.losses import get_loss
from lab.classes import ClassAdapter
from lab.model.model import Model

# py -m tensorboard.main --logdir=./lab/tensorboard_storage


class Main():
    def __init__(self,data,layers,loss):
        information = Information()
        data = get_data(data,information)
        layers = get_layers(layers,information)
        loss = get_loss(loss,information)
        classAdapter = ClassAdapter(information)
        model = Model(information)
        model.fit(data.data_train,1,1,0.3)
        information.to_tensorboard()



data = "number_classification"
layers = ["Reshape",[784],"Linear",[24],"Sigmoid","Linear",[10],"Sigmoid"]
loss = "MSELoss"
Main("number_classification",layers,loss)











