import numpy as np
from conv import Conv
from linear import Linear

class LeNet(object):
    def __init__(self):
        conv_filters = []
        conv_channels = []
        linear_nodes = []

        self.model = []
