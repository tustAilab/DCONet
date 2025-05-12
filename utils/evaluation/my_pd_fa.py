import numpy as np
import cv2
from skimage import measure


'''class my_PD_FA(object):
    def __init__(self, ):
        self.reset()

    def update(self, pred, label):
        max_pred= np.max(pred)
        max_label = np.max(label)
        pred = pred / np.max(pred) # normalize output to 0-1
        label = label.astype(np.uint8)

        # analysis target number
        num_labels, labels, _, centroids = cv2.connectedComponentsWithStats(label)
        #assert num_labels > 1
        if(num_labels <= 1):
            return

        # get masks and update background area and targets number
        back_mask = labels == 0
        tmp_back_area = np.sum(back_mask)
        self.background_area += tmp_back_area
        self.target_nums += (num_labels - 1)


        pred_binary = pred > 0.5

        # update false detection
        tmp_false_detect = np.sum(np.logical_and(back_mask, pred_binary))
        assert tmp_false_detect <= tmp_back_area
        self.false_detect += tmp_false_detect

        # update true detection, there maybe multiple targets
        for t in range(1, num_labels):
            target_mask = labels == t
            self.true_detect += np.sum(np.logical_and(target_mask, pred_binary)) > 0

    def get(self):
        FA = self.false_detect / self.background_area  #
        PD = self.true_detect / self.target_nums       #
        return PD,FA

    def get_all(self):
        return self.false_detect, self.background_area, self.true_detect, self.target_nums

    def reset(self):
        self.false_detect = 0
        self.true_detect = 0
        self.background_area = 0
        self.target_nums = 0
'''
class PD_FA():
    def __init__(self, nclass, bins,size):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
        self.size = size
    def update(self, preds, labels):
        # W = preds.shape[0]
        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins)
            predits  = np.array((preds > score_thresh)).astype('int64')
            predits = np.reshape(predits, (self.size, self.size))
            labelss = np.array((labels)).astype('int64')
            labelss = np.reshape(labelss, (self.size, self.size))
            # if W == 512:
            #     predits  = np.reshape (predits,  (512,512))#512
            #     labelss = np.array((labels).cpu()).astype('int64') # P
            #     labelss = np.reshape (labelss , (512,512))#512
            # elif W==384:
            #     predits = np.reshape(predits, (384, 384))  # 512
            #     labelss = np.array((labels).cpu()).astype('int64')  # P
            #     labelss = np.reshape(labelss, (384, 384))  # 512
            # else:
            #     predits = np.reshape(predits, (512//2, 512//2))  # 512
            #     labelss = np.array((labels).cpu()).astype('int64')  # P
            #     labelss = np.reshape(labelss, (512//2, 512//2))  # 512

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 2:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch)
            self.PD[iBin]+=len(self.distance_match)
        # print(len(self.image_area_total))
    def get(self,img_num):

        Final_FA =  self.FA / ((512*512) * img_num)#512
        Final_PD =  self.PD /self.target

        return Final_FA,Final_PD
    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])
