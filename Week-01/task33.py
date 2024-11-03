import numpy as np
import matplotlib.pyplot as plt

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
        
    return result_list
    
def visualize_randoms():
    list_range = np.arange(0, 101)
    list_of_randoms = random_numbers()
    
    plt.plot(list_range, list_of_randoms)
    plt.title("Random walk")
    plt.xlabel("Throw")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_randoms()