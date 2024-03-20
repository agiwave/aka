from .Bases import Input, Parameter
from .Engine import train, Trainer, load_weights, save_weights
from .Containers import *
from .Operators import *
from .Shapes import *
from .Activations import *
from .Others import *

class Args():
    def __init__(self, **kwargs): 
        for key in kwargs: setattr(self, key, kwargs[key])
