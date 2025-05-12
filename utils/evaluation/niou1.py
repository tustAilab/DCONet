import threading

import numpy
import numpy as np
import torch
import torch.nn.functional as F

# __all__ = ['SegmentationMetricTPFNFP']

def get_niou_prec_recall_fscore(niou):
    average_niou = np.mean(niou)

    return average_niou

class SegmentationMetricNIOU(object):
    """Computes pixAcc and mIoU metric scroes
    计算像素的准确度（pixacc和miou的评估）
    """

    def __init__(self, nclass):
        self.nclass = nclass #类别数
        self.lock = threading.Lock()#初始化锁，确保多线程情况下对共享数据的安全访问
        self.reset()# 调用reset函数初始化计数器
        self.niou = []
    def update(self, labels, preds):
        def evaluate_worker(self, label, pred):#子线程执行的任务，计算单个batch的TP、FP、FN
            tp, fp, fn = batch_tp_fp_fn(pred, label, self.nclass)
            with self.lock:
                niou = (1.0 * tp / (np.spacing(1) + tp + fp + fn))
                self.niou.append(niou)
            return

        if isinstance(preds, torch.Tensor):
            preds = (preds.detach().numpy() > 0).astype('int64')  # P
            labels = labels.numpy().astype('int64')  # T
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                        )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        #elif preds.dtype == numpy.uint8:
        elif isinstance(preds, np.ndarray):
            preds = ((preds / np.max(preds)) > 0.5).astype('int64')  # P
            labels = (labels / np.max(labels)).astype('int64')  # T
            evaluate_worker(self, labels, preds)
        else:
            raise NotImplemented

    def get_all(self):
        return self.total_tp, self.total_fp, self.total_fn

    def get(self):
        return get_niou_prec_recall_fscore(self.niou)

    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        return

def batch_tp_fp_fn(predict, target, nclass):
    """Batch Intersection of Union
    批量计算TP、FP和FN
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """

    mini = 1
    maxi = nclass
    nbins = nclass

    # predict = (output.detach().numpy() > 0).astype('int64')  # P
    # target = target.numpy().astype('int64')  # T
    intersection = predict * (predict == target)  # TP计算交集部分（TP）

    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))#预测的个数
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))#目标的个数

    # areas of TN FP FN
    area_tp = area_inter[0]
    area_fp = area_pred[0] - area_inter[0] #假阳性数量，即预测为第一个类别的数量减去真阳性数量.
    area_fn = area_lab[0] - area_inter[0]  #假阴性数量，即真实为第一个类别的数量减去真阳性数量.

    # area_union = area_pred + area_lab - area_inter
    assert area_tp <= (area_tp + area_fn + area_fp)
    return area_tp, area_fp, area_fn

