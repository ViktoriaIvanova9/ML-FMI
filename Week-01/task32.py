import numpy as np

def random_numbers():
    np.random.seed(123)
    
    result_list = []
    current_output = 0
    result_list.append(current_output)
    
    for _ in range(100):
        thrown_num = np.random.randint(1, 7)
        
        if thrown_num in {1,2}:
            thrown_num = -1
        elif thrown_num in {3,4,5}:
            thrown_num = 1
        else:
            thrown_num = np.random.randint(1, 7)
            
        if current_output + thrown_num >= 0:
            current_output += thrown_num
        result_list.append(current_output)
    
    print(result_list)

if __name__ == '__main__':
    random_numbers()