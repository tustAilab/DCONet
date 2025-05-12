import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
import torch.nn.functional as F

#from utils.metrics import ROCMetric
from utils.data import *

from PIL import Image
import os
import os.path as osp
import scipy.io as scio
import numpy as np

import cv2

from utils.evaluation.roc_cruve import ROCMetric
from utils.evaluation.my_pd_fa import PD_FA
#from utils.evaluation.niou import SamplewiseSigmoidMetric
from utils.evaluation.TPFNFP import SegmentationMetricTPFNFP
from utils.evaluation.niou1 import SegmentationMetricNIOU
# from utils.evaluation.niou2 import SamplewiseSigmoidMetric

class Dataset_mat(Data.Dataset):
    def __init__(self, dataset, base_size=256, thre=0.):
        
        self.base_size = base_size
        self.dataset = dataset
        if(dataset == 'NUDT-SIRST'):
            self.mat_dir = ''#.mat路径
            self.mask_dir = ''#label的路径
        elif(dataset == 'IRSTD-1K'):
            self.mat_dir  = ''
            self.mask_dir = ''
        elif(dataset == 'SIRST-aug'):
            self.mat_dir = ''
            self.mask_dir = ''
        else:
            raise NotImplementedError
        #获取.mat文件夹里面所有文件的名字  不含有扩展名字
        file_mat_names = os.listdir(self.mat_dir)
        self.file_names = [s[:-4] for s in file_mat_names]

        self.thre = thre #设置阈值  用于图像的二值化

        self.mat_transform = transforms.Resize((base_size, base_size), interpolation=Image.BILINEAR)
        self.mask_transform = transforms.Resize((base_size, base_size), interpolation=Image.NEAREST)

    def __getitem__(self, i):
        name = self.file_names[i]
        mask_path = osp.join(self.mask_dir, name) + ".png"
        mat_path = osp.join(self.mat_dir, name) + ".mat"

        #print(mask_path)

        rstImg = scio.loadmat(mat_path)['T']  #从mat里面读取变量T
        rstImg = np.asarray(rstImg) #将其转换为numpy的数组


        rst_seg = np.zeros(rstImg.shape)#创建一个和原图一样大小的数组 大于阈值的地方是1 其余是0
        rst_seg[rstImg > self.thre] = 1

        mask=cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), -1)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) #如果掩膜是三维的 转换为灰度图像
        mask = mask /mask.max()
        
        rstImg = cv2.resize(rstImg, dsize=(self.base_size, self.base_size), interpolation = cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.base_size, self.base_size), interpolation = cv2.INTER_NEAREST)

        return rstImg, mask

    def __len__(self):
        return len(self.file_names)


def cal_fpr_tpr(dataname, nbins=200, fileName = None):

    f = open(fileName, mode = 'a+')
    print('Running data: {:s}'.format(dataname))
    f.write('Running data: {:s}'.format(dataname) + '\n')
    baseSize = 256
    thre = 0.5
    pd_fa_instance=PD_FA(1,10,baseSize)
    pd_fa_instance.reset()
    #其他代码的方法
    # niou_metric =SamplewiseSigmoidMetric(nclass=1, score_thresh=0.5)
    # niou_metric.reset()
    dataset = Dataset_mat(dataname, base_size=baseSize, thre=thre)

    roc = ROCMetric(bins=200)
    #eval_PD_FA = my_PD_FA()
    eval_mIoU_P_R_F = SegmentationMetricTPFNFP(nclass=1)
    eval_niou=SegmentationMetricNIOU(nclass=1)
    for i in range(dataset.__len__()):
        rstImg, mask = dataset.__getitem__(i)

        size = rstImg.shape

        # print("rstImg shape:", rstImg.shape)
        # print("mask shape:", mask.shape)
        pd_fa_instance.update(preds=rstImg, labels=mask)
        #niou_metric.update(preds=rstImg, labels=mask)
        #eval_PD_FA.update(rstImg, mask)
        rstImg = torch.from_numpy(rstImg)  # Convert numpy array to torch tensor if necessary
        rstImg1 = F.sigmoid(rstImg)  # Apply sigmoid function on the tensor
        rstImg1[rstImg1 < 0] = 0
        rstImg1 = rstImg1.numpy().squeeze()

        roc.update(pred=rstImg1, label=mask)
        eval_mIoU_P_R_F.update(labels=mask, preds=rstImg1)
        eval_niou.update(labels=mask, preds=rstImg1)
    fpr, tpr, auc = roc.get()
    #pd, fa = eval_PD_FA.get()
    fa, pd = pd_fa_instance.get(dataset.__len__())
    miou, prec, recall, fscore = eval_mIoU_P_R_F.get()
    nIoU= eval_niou.get()
    print('AUC: %.6f' % (auc))
    f.write('AUC: %.6f' % (auc) + '\n')
    print('Pd: %.6f, Fa: %.8f' % (pd[0], fa[0]))
    f.write('Pd: %.6f, Fa: %.8f' % (pd[0], fa[0]) + '\n')
    print('mIoU: %.6f, Prec: %.6f, Recall: %.6f, fscore: %.6f' % (miou, prec, recall, fscore))
    f.write('mIoU: %.6f, Prec: %.6f, Recall: %.6f, fscore: %.6f' % (miou, prec, recall, fscore) + '\n')
    f.write('\n')
    print('niou: %.6f' % (nIoU))
    f.write('niou: %.6f' % (nIoU) + '\n')
    save_dict = {'tpr': tpr, 'fpr': fpr, 'Our Pd': pd[0], 'Our Fa': fa[0]}
    matDir = './eval/stage-1kmatResult/'
    if not os.path.exists(matDir):
        os.makedirs(matDir)
    matFile = osp.join(matDir, '{:s}.mat'.format(dataname))
    scio.savemat(matFile, save_dict)


if __name__ == '__main__':
    specific = True
    data_list = ['IRSTD-1K']

    fileDir = ('./eval/stage-1ktxtResult/')
    fileName = fileDir + 'mat_result.txt'
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)

    f = open(fileName, mode='w+')
    f.close()
    for data in data_list:
        cal_fpr_tpr(dataname=data, nbins=200, fileName = fileName)

