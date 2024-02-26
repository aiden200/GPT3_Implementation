import numpy
# import pandas
import math

'''
Small Tensor to calculate gradients
Inspired by Andrej Karpathy!
'''

class Mini_T:
    def __init__(self, value, _children=(), _op='') -> None:
        self.value = value
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def __repr__(self):
        return f"T(value={self.value})"
    
    def __add__(self, item):
        item = item if isinstance(item, Mini_T) else Mini_T(item) #change int into object T
        out = Mini_T(self.value + item.value, (self, item), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            item.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, item):
        item = item if isinstance(item, Mini_T) else Mini_T(item) #change int into object T
        out = Mini_T(self.value * item.value, (self, item), "*")

        def _backward():
            self.grad += item.value * out.grad
            item.grad += self.value * out.grad
        out._backward = _backward

        return out
    

    def __rmul__(self, item): # item *self
        return self * item
    
    def __radd__(self, item):
        return self + item
    
    def __neg__(self): # -self
        return self * -1
    
    def __sub__(self, item):
        return self + (-item)

    def __rsub__(self, item):
        return item + -(self)

    def __pow__(self, item):
        assert isinstance(item, (int, float)), "Only supporting int/float"

        out = Mini_T(self.value ** item, (self,), f"**{item}")

        def _backward():
            self.grad += item * self.value ** (item-1)
        
        out._backward = _backward

        return out
    
    def __truediv__(self, item): #self/item
        return self * item**-1

    def tanh(self):
        x = (math.exp(2*self.value) -1 )/ (math.exp(2*self.value) +1 )
        out = Mini_T(x, (self,))

        def _backward():
            self.grad += (1-x**2) * out.grad
        out._backward = _backward

        return out 
    
    def exp(self):
        x = self.value
        out = Mini_T(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad = out.value  * out.grad
        out._backward = _backward

        return out

    def backward(self):
        g = []
        visited = set()
        def build_g(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_g(child)
                g.append(v)
        build_g(self)
        self.grad = 1.0

        for node in reversed(g):
            node._backward()

import random

class Neuron:
    def __init__(self, nin):
        self.w = [Mini_T(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Mini_T(random.uniform(-1,1))
    
    def __call__(self, x):
        
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:
    def __init__(self, nin, nouts ):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

n = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]

ys = [1.0, -1.0, -1.0, 1.0]
ypred = [n(x) for x in xs]
print(ypred)

loss = sum((y-yhat)**2 for y, yhat in zip(ys, ypred))
print(f"Loss: {loss}")

print(n.layers[0].neurons[0].w[0].grad)
loss.backward()
print(n.layers[0].neurons[0].w[0].grad)


