import graphviz
import numpy as np

class Value:
    def __init__(self, data : float, _op='', label='', grad=0.0000):
        self.data = data
        self._prev = set()
        self._op = _op
        self.label = label
        self.grad = grad
    
    def __add__(self, other):
        res = Value(self.data + other.data, '+')
        res._prev = {self, other}
        return res
    
    def __mul__(self, other):
        res = Value(self.data * other.data, '*')
        res._prev = {self, other}
        
        return res
    
    def __repr__(self):
        return (f'Value(data={self.data})')
    

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

def topological_sort(topological_order, node, visited):
    if node not in visited:
        visited.add(node)
        for parent in node._prev:
            topological_sort(topological_order, parent, visited)
        topological_order.append(node)

def manual_der(node):
    node.grad = 1.0
    visited = set()
    topological_order = []
    topological_sort(topological_order, node, visited)

    for node in reversed(topological_order):
        if node._op == '+':
            a, b = node._prev
            a.grad += node.grad
            b.grad += node.grad
        
        elif node._op == '*':
            a, b = node._prev
            a.grad += b.data * node.grad
            b.grad += a.data * node.grad

def draw_dot(root: Value) -> graphviz.Digraph:
    dot = graphviz.Digraph(filename='01_result', format='svg', graph_attr={
                           'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    name = 'a'
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node
        dot.node(name=uid, label=f'{{{name} | data: {n.data} | grad: {n.grad} }}', shape='record')
        name = chr(ord(name) + 1)
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
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    d = Value(-2.0, label='d')
    e = a * b
    e.label = 'e'
    f = e + c
    f.label = 'f'
    result = f * d
    result.label = 'L'

    manual_der(result)
    draw_dot(result).render(directory='./graphviz_output', view=True)

    print(f'Old L = {result.data}')

    a.data += 0.01 * a.grad
    b.data += 0.01 * b.grad
    c.data += 0.01 * c.grad
    d.data += 0.01 * d.grad

    new_result = (a * b + c) * d
    print(f'New L = {new_result.data}')

if __name__ == '__main__':
    main()