import numpy as np
from nn_functions import ReLU

class MLP:

    def __init__(self) -> None:
        self.layers: list[Layer] = []
    
    def addLayer(self, layer: 'Layer') -> None:
        self.layers.append(layer)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

class Layer:

    def __init__(self, d_in: int, d_out) -> None:
        self.weights = np.random.rand(d_in, d_out)
        self.bias = np.random.rand(d_out)
        self.activation = ReLU
    
    def forward(self, input: np.ndarray) -> np.ndarray:
        print(input.shape, self.weights.shape)
        return self.activation(np.matmul(input, self.weights) + self.bias)