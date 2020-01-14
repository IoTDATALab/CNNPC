import cnnpc_main_T0 as cnnpc_for_acc
import cnnpc_main_A0 as cnnpc_for_t
from src_cnnpc.tools import partitions_init, set_model
import os
from options import args_parser
    
if __name__ == '__main__':
    
    args = args_parser()

    model = args.model
    end_device = args.end_device
    bandwidth = args.bandwidth

    set_model(model) # create set.txt
    partitions_init(model)
    os.system('cp ./R-C-Time/'+model+'/TR-'+end_device+'/T_R.npy ./model_profile/T_R.npy')
    os.system('cp ./R-C-Time/'+model+'/TC-20M-'+bandwidth+'M/T_C.npy ./model_profile/T_C.npy')

    if args.A0:
        cnnpc_for_t.main(args.A0)
        os.system('mv ./process.txt ./result/'+model+'/'+end_device+'-20M-'+bandwidth+'M/process-'+str(args.A0)+'.txt')
        os.system('mv ./result.txt ./result/'+model+'/'+end_device+'-20M-'+bandwidth+'M/result-'+str(args.A0)+'.txt')        

    if args.T0:
        cnnpc_for_t.main(args.T0)
        os.system('mv ./process.txt ./result/'+model+'/'+end_device+'-20M-'+bandwidth+'M/process-'+str(args.T0)+'.txt')
        os.system('mv ./result.txt ./result/'+model+'/'+end_device+'-20M-'+bandwidth+'M/result-'+str(args.T0)+'.txt')        




