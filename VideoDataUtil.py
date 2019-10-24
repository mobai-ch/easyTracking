import cv2
import numpy
import os

OTBDir = "E:/CVCode/Tracking/OTB100/OTB100/"

class OTBTool:
    def __init__(self, dir=OTBDir):
        self.videos = os.listdir(dir)
    def getOneVideo(self, num):
        dir = OTBDir + self.videos[num]
        imglist = [dir+'/img/'+imgname for imgname in os.listdir(dir+'/img/')]
        ground_truth = [x.strip().split(',') for x in open(dir+'/groundtruth_rect.txt', 'r').readlines()]
        for i in range(len(imglist)):
            ground_truth[i] = [int(x) for x in ground_truth[i]]
        return imglist, ground_truth
