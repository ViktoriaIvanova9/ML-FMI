import numpy as np

class Value:
    def __init__(self, fnum : float):
        self.fnum = fnum

    def __repr__(self):
        return (f'Value(data={self.fnum})')
    
    def __add__(self, other):
        return Value(self.fnum + other.fnum)

def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    result = x + y
    print(result)

if __name__ == '__main__':
    main()