import numpy as np

class Value:
    def __init__(self, fnum : float):
        self.fnum = fnum
        self._prev = set()

    
    def __add__(self, other):
        res = Value(self.fnum + other.fnum)
        res._prev = {self, other}
        return res
    
    def __mul__(self, other):
        res = Value(self.fnum * other.fnum)
        res._prev = {self, other}
        return res
    
    def __repr__(self):
        return (f'Value(data={self.fnum})')

def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z
    print(result._prev)

if __name__ == '__main__':
    main()

# class Value:
#     def __init__(self, data: float):
#         self.data = data  # The stored floating point number
#         self._prev = set()  # Holds the values that produced the current value

#     def __add__(self, other):
#         result = Value(self.data + other.data)
#         result._prev = {self, other}
#         return result

#     def __mul__(self, other):
#         result = Value(self.data * other.data)
#         result._prev = {self, other}
#         return result

#     def __repr__(self):
#         return f"Value(data={self.data})"

# def main() -> None:
#     x = Value(2.0)
#     y = Value(-3.0)
#     z = Value(10.0)
#     result = x * y + z
#     print(result._prev)

# if __name__ == "__main__":
#     main()
