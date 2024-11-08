import numpy as np
import matplotlib.pyplot as plt

def random_numbers():
    
    result_list = []
    current_output = 0
    result_list.append(current_output)
    
    for _ in range(100):
        if np.random.rand() <= 0.005:
            current_output = 0
            result_list.append(current_output)
            continue

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
    
def repeat_randoms():
    np.random.seed(123)
    all_walks = []
    
    for _ in range(20):
        all_walks.append(random_numbers())
        
    return all_walks

def visualize_walks():
    np_all_walks = repeat_randoms()
    list_range = np.arange(0, 101)
    
    for elem in np_all_walks:
        plt.plot(list_range, elem)
        plt.title("Random walks")
        plt.xlabel("Throw")

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_walks()