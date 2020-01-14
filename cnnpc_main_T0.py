import numpy as np
import src_cnnpc.CRS as CRS
from src_cnnpc.R_support import create_R, update_R, nextPoint_R, get_R, update_R_CAE
from src_cnnpc.latency_support import min_profile_T, get_T, cnnpc_latency_range 
from src_cnnpc.tools import add_logs, turn_r_L

def main(input_T, is_warmup=False):
    '''main function

    Args:
    * input_T: the latency requirement
    '''
    T_range = cnnpc_latency_range()
    if input_T < T_range[0] or input_T > T_range[1]:
        print('Error: Improper input_T')
        return 0
        
    T0 = input_T
    A = 0
    L_partition = np.load("./model_profile/partitions.npy")# load the partition points and their channel amount in model：[[1, 11, ...],[64, 64, ...],[2,22,...]]
    num_partition = L_partition.shape[1]
    rate_temp = [0.0, 0.0]  # current optimal compression rates
    L_temp = [0, 0]  # current optimal partition points
    warmup = is_warmup
    if not warmup:
        create_R()
    add_logs('T0 is:' + str(T0))

    for i in range(num_partition):
        add_logs("------Partition Layer " + str(i) + " Search Begin------\n")
        # start to single device search
        R = np.zeros((L_partition[1][i]), dtype=np.float32)# create solution space R for single layer compression
        rate_min = 1.0
        for m in range(L_partition[1][i]):
            R[m] = m / L_partition[1][i]
            if get_T([i], [R[m]]) <= T0 and R[m] < rate_min :
                rate_min = R[m]

        if rate_min < 1.0:
            a = CRS.A(i, rate_min)
        if a > A:
            L_temp = [i, 0]
            rate_temp = [rate_min, 0.0]
            A = a
            logs = "------Best point changed!------\n"
            logs += "New Best Partitions: " + str(turn_r_L(L_temp)) + "\n"
            logs += "New Best Compression Rate: " + str(rate_temp) + "\n"
            logs += "New Best Accuracy: " + str(A) + "\n"
            add_logs(logs)

        if rate_min == 0.0:
            break

        for j in range(i+1, num_partition):
            add_logs("For partition Layer " + str(j) + " Search\n")
            update_R(i, rate_min, T0, j, 0) # update R
            while True:
                next_point = nextPoint_R(i, j)   
                if not next_point:
                    break
                rate_1 = next_point[0]
                rate_2 = next_point[1]       
                add_logs('next_point :' + str(next_point) + "\n") 
                add_logs("------For judgment：------" + "\n")
                IsUpdate_rate1 = True
                if CRS.CAE(j, rate_2, j, rate_2) > A and CRS.A(j, rate_2) > A and CRS.CAE(i, rate_1, j, rate_2) > A :
                    A_ = CRS.A(j, rate_2, i, rate_1)
                    if A_ > A : # update the optimal solution
                        rate_temp = [rate_1 ,rate_2]
                        L_temp = [i , j]
                        A = A_
                        logs = "------Best point changed!------\n"
                        logs += "New Best Partitions: " + str(turn_r_L(L_temp)) + "\n"
                        logs += "New Best Compression Rate: " + str(rate_temp) + "\n"
                        logs += "New Best Accuracy: " + str(A) + "\n"
                        add_logs(logs)
                        IsUpdate_rate1 = False
                add_logs('For update_R_CAE\n')
                update_R_CAE(i, j, rate_2, A, IsUpdate_rate1=IsUpdate_rate1, rate1=rate_1)  # update solution space R 

    add_logs("Best point:" + str(turn_r_L(L_temp)) + " " + str(rate_temp) + " " +str(A) + "\n")

    if L_temp[0] >= L_temp[1] and rate_temp[0] >= rate_temp[1]: # get the number of compression layer
        all_latency, best_deploy = get_T([L_temp[0]], [rate_temp[0]], for_minT=False)
    else:
        all_latency, best_deploy = get_T(L_temp, rate_temp, for_minT=False)
    add_logs(all_latency)

    result = "When T0 is " + str(T0) + "\n"
    result += "Best point:" + str(turn_r_L(L_temp)) + " " + str(rate_temp) + " " +str(A) + "\n"
    result += "The deployment way:" + best_deploy
    with open('result.txt', 'w+') as f:
        f.write(result)

if __name__ == '__main__':  
    pass
    ## Rsnet
    # 80.595775 62.257 43.917235 Mi8 20M 10M 
    # 81.208275 63.48155 45.754825 Mi8 20M 5M 
    # 83.552675 68.17035 52.788025 Mi8 20M 1M

    # 83.1324 67.3298 51.5272 MI8SE 20M 10M
    # 83.7449 68.5548 53.3647 MI8SE 20M 5M
    # 86.0893 73.2346 60.3979 MI8SE 20M 1M

    ## Mobilenet
    # 66.133125 52.18065 38.228175 Mi 20M 10M 
    # 66.1750   52.2554  38.3403   Mi 20M 5M
    # 66.469575 52.85355 39.237525 Mi 20M 1M

    # 70.14935  55.87134 41.59332  Mi 20M 10M
    # 70.186735 55.94609 41.705445 Mi 20M 5M
    # 70.485581 56.54424 42.60267  Mi 20M 1M
