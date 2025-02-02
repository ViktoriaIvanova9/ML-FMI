import numpy as np

class Value:
    def __init__(self, data : float):
        self.data = data

    def __repr__(self):
        return (f'Value(data={self.data})')
    
    def __add__(self, other):
        return Value(self.data + other.data)
    
    def __mul__(self, other):
        return Value(self.data * other.data)

def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result)

if __name__ == '__main__':
    main()