from torch import no_grad
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import shutil
from lab.memory.memory import Memory
memory = Memory()
# Path to the directory
library_path = './lab/tensorboard_storage/Run'



# default `log_dir` is "runs" - we'll be more specific here
#  py -m tensorboard.main --logdir=./lab/tensorboard_storage
# https://pytorch.org/docs/stable/tensorboard.html

def count():
    c = memory.data["count"]+1
    memory.store(("count",c))
    return c

def clear_library():
    # Ensure the path exists
    if os.path.exists(library_path):
        # Iterate over the files and subdirectories in the directory
        for filename in os.listdir(library_path):
            file_path = os.path.join(library_path, filename)
            try:
                # Check if it's a file or a directory and delete accordingly
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or symlink
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directory
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        print("All files and subdirectories deleted successfully.")
    else:
        print(f"The directory {library_path} does not exist.")



def to_tensorboard(input:[]):
    # clear_library()
    """

    :param input:

    writing format :
    for model :
    self.tensorboard["graph"].append((self:model,self.input: tensor[input size]))

    for image:
    self.tensorboard["image"].append((name:str, img_grid:tensor[RGB[3],X,Y]))
    :return:




    """
    writer = SummaryWriter(library_path)
    for o in input:
        add_object(o,writer)
    writer.close()

def add_object(o,writer):
    tensor_board = o.to_tensorboard()
    tensor_board_add_map = {
        "image":writer.add_image,
        "graph":writer.add_graph,
        "scalar":writer.add_scalar

    }
    for type , value in tensor_board.items():
        for item in value:
            with torch.no_grad():
                if type in ["scalar"]:
                    name,xy = item
                    c = count()
                    for y,x in xy.items():
                        tensor_board_add_map[type](name+f" {c}",x,y)

                else:
                    tensor_board_add_map[type](*item)



