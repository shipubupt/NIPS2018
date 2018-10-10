import os
import scipy.io as sio
seqs = sio.loadmat('OTB50.mat')
#seqs = sio.loadmat('OTB51-100.mat')
for i in range(len(seqs['seqs'][0])):
    name = str(seqs['seqs'][0][i][0][0][0][0])
    if name != "football":

        continue
    print name
    if os.path.exists("../result_labgpu_test/"+name+"/result.json"):
        print name
        continue

    os.system("/home/lxq/anaconda2/envs/ps_pytorch/bin/python run_tracker.py -s "+name)

