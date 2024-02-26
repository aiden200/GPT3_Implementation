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
    
    def __neg__(self): # -self
        return self * -1
    
    def __sub__(self, item):
        return self + -item

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


##Checking
    
a = Mini_T(3)
b = Mini_T(4.0)
c = Mini_T(-3)

print(a * b + c)

o = a * b  +c
o.backward()
print(a.grad)
