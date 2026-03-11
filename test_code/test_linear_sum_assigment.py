import numpy as np
import sys
from scipy.optimize import linear_sum_assignment
if __name__ == '__main__':
    cost = np.array([[1,0,0],[0,1,0],[0,0,0]])
    row_ind, col_ind = linear_sum_assignment(cost, True)
    print(f'row_ind: {row_ind}')
    print(f'col_ind: {col_ind}')