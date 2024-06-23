import numpy as np
import nn

def ReLU(x: nn.Tensor) -> nn.Tensor:
    return nn.Tensor(np.maximum(np.zeros(x.val.shape), x.val))