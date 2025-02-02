import graphviz
import numpy as np

class Value:
    def __init__(self, data : float, _op='', label='', grad=0.0000, _backward = lambda : None):
        self.data = data
        self._prev = set()
        self._op = _op
        self.label = label
        self.grad = grad
        self._backward = _backward
    
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        res = Value(self.data + other.data, '+')
        res._prev = {self, other}

        def backward():
            self.grad += res.grad
            other.grad += res.grad

        res._backward = backward
        return res
    
    def __radd__(self, other):
        res = self + other
        return res
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        res = Value(self.data * other.data, '*')
        res._prev = {self, other}

        def backward():
            self.grad += other.data * res.grad
            other.grad += self.data * res.grad            

        res._backward = backward
        return res
    
    def __rmul__(self, other):
        res = self * other
        return res
    
    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        res = Value(self.data / other.data, '/')
        res._prev = {self, other}

        def backward():
            self.grad += (1 / other.data) * res.grad # der by self
            other.grad += (-self.data / (other.data ** 2)) * res.grad # der by other       
        res._backward = backward

        return res

    def __pow__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        res = Value(self.data ** other.data, f'**{other.data}')
        res._prev = {self}

        def backward(): # x^3 = 3 x^2
            self.grad += (other.data * (self.data ** (other.data - 1))) * res.grad

        res._backward = backward
        return res
    
    def exp(self):
        res = Value(np.exp(self.data), 'exp')
        res._prev = {self}

        def backward():
            self.grad += res.data * res.grad

        res._backward = backward
        return res
    
    def tanh(self):
        res = Value(((np.exp(2*self.data) - 1) / (np.exp(2*self.data) + 1)), 'tanh')
        res._prev = {self}

        def backward():
            self.grad += (1 - res.data * res.data) * res.grad

        res._backward = backward
        return res

    def __repr__(self):
        return (f'Value(data={self.data})')
    
    def backward(self):
        self.grad = 1.0
        visited = set()
        topological_order = []
        self.topological_sort(topological_order, visited)

        for node in reversed(topological_order):
            node._backward()

    def topological_sort(self, topological_order, visited):
        if self not in visited:
            visited.add(self)
            for parent in self._prev:
                parent.topological_sort(topological_order, visited)
            topological_order.append(self)


def trace(obj):
    nodes = set()
    edges = set()

    trace_rec_adding(obj, nodes, edges)
    return nodes, edges

def trace_rec_adding(obj, nodes, edges):
    nodes.add(obj)

    for prev in obj._prev:
        edges.add((prev, obj))
        trace_rec_adding(prev, nodes, edges)


def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(name=uid, label=f'{{{n.label} | data: {np.round(n.data, 4)} | grad: {np.round(n.grad, 4)} }}', shape='record')
        if n._op:
            # if this value is a result of some operation, create an "op" node for the operation
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to the node of the operation
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the "op" node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def main() -> None:
    x1 = Value(2.0, label='x1')
    x2 = Value(0.0, label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0, label='w2')
    b = Value(6.8813735870195432, label='b')

    x1w1 = x1 * w1
    x1w1.label = 'x1w1'

    x2w2 = x2 * w2
    x2w2.label = 'x2w2'

    sum_weights = x1w1 + x2w2
    sum_weights.label = 'x1w1 + x2w2'

    logit = sum_weights + b
    logit.label = 'logit'

    two = Value(2.0, label='2')
    one = Value(1.0, label='1')
    minus_one = Value(-1.0, label='-1')

    result = two * logit

    exp_result = np.exp(result)
    exp_result.label = 'e'

    result_plus_one = exp_result + one
    result_minus_one = exp_result + minus_one
    
    result_pow_minus_one = result_plus_one ** minus_one

    L = result_minus_one * result_pow_minus_one
    L.label = 'L'
    
    L.backward()
    draw_dot(L).render(directory='./graphviz_output', view=True)
    
if __name__ == '__main__':
    main()