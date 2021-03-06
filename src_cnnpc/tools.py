import numpy as np
import os

def add_logs(lines):
    '''add logs to file
    
    Args:
    * lines: the content will be recorded in process.txt
    '''
    with open('process.txt', 'a+') as f:
        f.write(lines)
    print(lines)


def get_net_name():
    f = open('set.txt','r')
    init_set = f.readline()
    f.close()
    net_name = init_set.rstrip('\n')
    return  net_name


def set_model(model):
    os.system('cp -r ./learners/channel_pruning_'+model+'/* ./learners/channel_pruning')
    with open('set.txt', 'w+') as f:
        f.write(model)


def partitions_init(model):

    resnet_partitions = [[1,  6,  11,  16],
                         [64, 64, 128, 256],
                         [1,  6,  11,  16]] # [3,  13, 23,  33]]

    mobilenet_partitions = [[0,  1,  3,  6,  10, 13,  16],
                            [32, 16, 24, 32, 64, 96, 160],
                            [1,  2,  6,  12, 20, 26,  32]]

    # ssd_partitions = [[],
    #                   [],
    #                   []]

    if model == 'resnet':
        np.save('./model_profile/partitions.npy', np.asarray(resnet_partitions))

    if model == 'mobilenet':
        np.save('./model_profile/partitions.npy', np.asarray(mobilenet_partitions))


def turn_r_L(L):

    L_partition = np.load("./model_profile/partitions.npy")

    r_L0 = L_partition[0][L[0]]
    r_L1 = L_partition[0][L[1]]

    return [r_L0, r_L1]