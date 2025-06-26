from .DCONet import *


def get_model(name, net=None):
    if name == 'rpcanet':
        net = DCONet(stage_num=5)
    else:
        raise NotImplementedError

    return net

