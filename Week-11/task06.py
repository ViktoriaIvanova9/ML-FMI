import numpy as np

class Value:
    def __init__(self, data : float, _op=''):
        self.data = data
        self._prev = set()
        self._op = _op
    
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


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    
    nodes, edges = trace(x)
    print('x')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(y)
    print('y')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(z)
    print('z')
    print(f'{nodes=}')
    print(f'{edges=}')
    
    nodes, edges = trace(result)
    print('result')
    print(f'{nodes=}')
    print(f'{edges=}')

if __name__ == '__main__':
    main()

# x
# nodes={Value(data=2.0)}
# edges=set()
# y
# nodes={Value(data=-3.0)}
# edges=set()
# z
# nodes={Value(data=10.0)}
# edges=set()
# result
# nodes={Value(data=10.0), Value(data=-3.0), Value(data=4.0), Value(data=-6.0), Value(data=2.0)}
# edges={(Value(data=-6.0), Value(data=4.0)), (Value(data=10.0), Value(data=4.0)), (Value(data=-3.0), Value(data=-6.0)), (Value(data=2.0), Value(data=-6.0))}