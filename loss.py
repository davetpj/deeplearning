import numpy as np

# SSE
def sum_squares_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# CEE
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))



    
if __name__=='__main__':
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    
    # case _1 
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    result = sum_squares_error(np.array(y), np.array(t))
    print(result)
    result = cross_entropy_error(np.array(y), np.array(t))
    print(result)
    
    # case _2
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    result = sum_squares_error(np.array(y), np.array(t))
    print(result)
    result = cross_entropy_error(np.array(y), np.array(t))
    print(result)