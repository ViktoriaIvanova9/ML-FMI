import numpy as np

def baseball_numpy_array():
    baseball = [180, 215, 210, 210, 188, 176, 209, 200]
    numpy_baseball = np.array(baseball)
    print(f'Baseball array: {numpy_baseball}')
    print(f'Type of baseball array: {type(numpy_baseball)}')

if __name__ == '__main__':
    baseball_numpy_array()
