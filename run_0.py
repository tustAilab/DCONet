import os
os.environ['CUDA_VISIBLE_DEVICE'] ='0'
gpu = '--gpu 0'

train_cfg = ' --epoch 400 --lr 1e-4 --save-iter-step 100 --log-per-iter 5 '
data_sirstaug = ' --dataset sirstaug '
data_irstd1k = ' --dataset irstd1k '
data_nudt = ' --dataset nudt '

for i in range(1):
    os.system('python train.py --net-name rpcanet --batch-size 8' + train_cfg + data_irstd1k  + gpu)


