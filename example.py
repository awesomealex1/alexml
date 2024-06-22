import nn
import numpy as np

net = nn.MLP()

layer1 = nn.Layer(20, 40)
layer2 = nn.Layer(40, 1)

net.addLayer(layer1)
net.addLayer(layer2)

x = np.ones((1,20))
y = net.forward(x)

print(y)