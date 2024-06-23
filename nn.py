import numpy as np
from nn_functions import ReLU

class Tensor:

    def __init__(self, x: np.ndarray) -> None:
        self.val = x

class MLP:

    def __init__(self) -> None:
        self.layers: list[Layer] = []
    
    def addLayer(self, layer: 'Layer') -> None:
        self.layers.append(layer)
    
    def forward(self, x: 'Tensor') -> 'Tensor':
        for layer in self.layers:
            x = layer.forward(x)
        return x

class Layer:

    def __init__(self, d_in: int, d_out) -> None:
        self.weights = Tensor(np.random.rand(d_in, d_out))
        self.bias = Tensor(np.random.rand(d_out))
        self.activation = ReLU
    
    def forward(self, input: 'Tensor') -> 'Tensor':
        return self.activation(np.matmul(input.val, self.weights.val) + self.bias.val)