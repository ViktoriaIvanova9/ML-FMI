import numpy as np

def random_numbers():
    np.random.seed(123)
    

    print(f'Random float: {np.random.rand()}')
    print(f'Random integer 1: {np.random.randint(1, 7)}')
    print(f'Random integer 2: {np.random.randint(1, 7)}')
    
    current_output = 50
    print(f'Before throw step = {current_output}')
    
    thrown_num = np.random.randint(1, 7)
    print(f'After throw dice = {thrown_num}')
    
    if thrown_num in {1,2}:
        thrown_num = -1
    elif thrown_num in {3,4,5}:
        thrown_num = 1
    else:
        thrown_num = np.random.randint(1, 7)
        
    current_output += thrown_num
    print(f'After throw step = {current_output}')

if __name__ == '__main__':
    random_numbers()