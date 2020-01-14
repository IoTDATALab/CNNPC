import numpy as np
import src_cnnpc.CRS as CRS
from src_cnnpc.latency_support import get_T

def create_R():
    '''create solution space '''
    a = np.load("./model_profile/partitions.npy")
    num = a.shape[1]
    for i in range(0, num):
        for j in range(i+1, num):
            b = np.ones((a[1][i], a[1][j]), dtype=np.bool)
            name = './R/' + str(i) + 'and' + str(j) + '.npy'
            np.save(name, b)    


def update_R(com1, rate1, T, com2, flag):
    '''update the solution space R

    Args:
    * com1,com2: denote the compression layer
    * rate1: denote the compressino rate
    * T: current best latency
    * flag: choose the method to update R
    '''
    filename = "./R/" + str(com1) + "and" + str(com2) + ".npy"
    R = np.load(filename)       
    n_1 = round(rate1 * R.shape[0]) # the number of residual channel at com1 layer

    if flag:
        for i in range(1, R.shape[0]):
            for j in range(1, R.shape[1]):
                if (i < n_1 and get_T([com1, com2], [i / R.shape[0], j / R.shape[1]]) < T):
                    pass
                else:
                    R[i][j] = False
    else:
        for i in range(1, R.shape[0]):
            for j in range(1, R.shape[1]):
                if (i <= n_1 and get_T([com1, com2], [i / R.shape[0], j / R.shape[1]]) < T):
                    pass
                else:
                    R[i][j] = False        

    np.save(filename, R)

def update_R_CAE(com1, com2, rate2, A, IsUpdate_rate1=True, rate1=0.0):
    '''update the solution space R 

    Args:
    * com1,com2: denote the compression layer
    * rate1: denote the compressino rate
    * A: current best accuracy
    '''
    filename = "./R/" + str(com1) + "and" + str(com2) + ".npy"
    R = np.load(filename)       
    n_2 = round(rate2 * R.shape[1]) # the number of residual channel at com2 layer
    n_1 = 0
    if IsUpdate_rate1:
        n_1 = round(rate1 * R.shape[0])
    for i in range(1, R.shape[0]):
        for j in range(1, R.shape[1]):
            if R[i][j]:
                if IsUpdate_rate1:
                    if(i > n_1 and j > n_2 and CRS.CAE(com1, i / R.shape[0], com2, j / R.shape[1]) > A):
                        pass
                    else:
                        R[i][j] = False                    
                else:
                    if(j > n_2 and CRS.CAE(com1, i / R.shape[0], com2, j / R.shape[1]) > A):
                        pass
                    else:
                        R[i][j] = False
    np.save(filename, R)

def nextPoint_R(com1, com2):
    '''output the next compression rates according to following rules:
        a2 ← min a2 in a ∈ R
        a1 ← min a1 in a (·, a2) ∈ R

    Args:
    * com1, com2: denote the compression layer

    Returns:
    [a1, a2]: the next compression rates
    '''
    filename = "./R/" + str(com1) + "and" + str(com2) + ".npy"
    R = np.load(filename)

    n_2 = 1000
    for i in range(1, R.shape[0]):
        for j in range(1, R.shape[1]):
            if R[i][j]:
                if (j < n_2):
                    n_2 = j

    if (n_2 == 1000):
        return []

    n_1 = 1000
    for i in range(1, R.shape[0]):
        if R[i][n_2]:
            if (i < n_1):
                n_1 = i

    if (n_1 == 1000):
        return []

    rate1 = n_1 / R.shape[0]
    rate2 = n_2 / R.shape[1]

    return [rate1, rate2]

def get_R(com1, rate1, com2):
    '''get the partial solution space according to the [com1, rate1, com2]

    Args:
    * com1, com2: denote the compression layer
    * rate1: denotes the compression rate

    Returns:
    * r: the partial solution space
    '''
    r = []
    filename = "./R/" + str(com1) + "and" + str(com2) + ".npy"
    R = np.load(filename)      
    n_1 = round(rate1 * R.shape[0])
    for j in range(1, R.shape[1]):
        if R[n_1][j]:
            r.append(j / R.shape[1])

    r = np.array(r, dtype=np.float32)

    return r
