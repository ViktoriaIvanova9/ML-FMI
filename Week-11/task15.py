import graphviz
import numpy as np

class Value:
    def __init__(self, data : float, _op='', label='', grad=0.0000, _backward = lambda : None): # is there another way to initialise 
                                                                                                # the function
        self.data = data
        self._prev = set()
        self._op = _op
        self.label = label
        self.grad = grad
        self._backward = _backward
    
    def __add__(self, other):
        res = Value(self.data + other.data, '+')
        res._prev = {self, other}

        def backward():
            self.grad += res.grad
            other.grad += res.grad

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
        dot.node(name=uid, label=f'{{{n.label} | data: {n.data} | grad: {n.grad} }}', shape='record')
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
    x = Value(10.0, label='x')
    y = Value(5.0, label='y')
    z = x + y
    z.label = 'z'
    
    # This will create a new directory and store the output file there.
    # With "view=True" it'll automatically display the saved file.
    z.backward()
    draw_dot(z).render(directory='./graphviz_output', view=True)

if __name__ == '__main__':
    main()
