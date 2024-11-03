import numpy as np
import matplotlib.pyplot as plt

def random_numbers_ends():
    
    result_list = []
    current_output = 0
    result_list.append(current_output)
    
    for _ in range(100):
        if np.random.rand() <= 0.005:
            current_output = 0

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
        
    return result_list[-1]
    
def repeat_randoms():
    np.random.seed(123)
    all_walks_ends = []
    
    for _ in range(500):
        all_walks_ends.append(random_numbers_ends())
        
    return all_walks_ends

def visualize_walks():
    np_all_walks = np.array(repeat_randoms())
    
    plt.hist(np_all_walks, bins=10)
    
    plt.title("Random walks")
    plt.xlabel("End step")
    
    # print(len(np_all_walks[np_all_walks > 60])/500)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_walks()
    
    # Ok, so what are the odds that you'll reach 60 steps high on the Empire State Building? - 60 %