from torch import no_grad
from torch.utils.tensorboard import SummaryWriter
import torch

# default `log_dir` is "runs" - we'll be more specific here
#  py -m tensorboard.main --logdir=./lab/tensorboard_storage


def to_tensorboard(input:[]):
    writer = SummaryWriter('./lab/tensorboard_storage')
    for o in input:
        add_object(o,writer)
    writer.close()

def add_object(o,writer):
    tensor_board = o.to_tensorboard()
    tensor_board_add_map = {
        "image":writer.add_image,
        "graph":writer.add_graph
    }
    for type , value in tensor_board.items():
        for item in value:
            with torch.no_grad():
                tensor_board_add_map[type](*item)



