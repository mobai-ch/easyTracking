'''
再这个文件中，我们尝试去用滑动窗口（传统艺能）实现一个检测器，
毕竟这次就是做一些比较复古的事情，这个和TLD不太一样, 我是仅仅使用了直方图+滑动窗口+SVM。
'''

import cv2
import numpy as np
from sklearn.svm import SVR
from VideoDataUtil import OTBTool
import matplotlib.pyplot as plt

class Detector:
    def __init__(self):
        self.clf = SVR(gamma='auto')
        self.Plabel = []
        self.NLabel = []
        self.tarW = 0
        self.tarH = 0

    def initDetector(self, img, rect):
        img_width = img.shape[1]
        img_height = img.shape[0]
        Phists = []
        Nhists = []

        [x, y, w, h] = rect
        self.tarW = w
        self.tarH = h

        for i in range(-5, 5):
            for j in range(-5, 5):
                inRect = (int(x-i*0.02*w), int(y-j*0.02*h), w, h)
                p1 = (inRect[0], inRect[1])
                p2 = (inRect[0]+w, inRect[1]+h)
                DetectOne = img[p1[1]:p2[1],p1[0]:p2[0]]
                histOne = np.bincount(DetectOne.ravel(), minlength=256)
                histOne = histOne/np.sum(histOne)
                Phists.append(histOne)
        
        for i in range(0, int((img_width-w)/w)+2):
            for j in range(0, int((img_height-h)/h)+2):
                leftx = w*i
                lefty = h*j
                if leftx > x-0.8*w and leftx < x+0.8*w and lefty > y-0.8*h and lefty < y+0.8*h:
                    pass
                else:
                    NRect = img[lefty:lefty+h, leftx:leftx+w]
                    histOne = np.bincount(NRect.ravel(), minlength=256)
                    histOne = histOne/np.sum(histOne)
                    Nhists.append(histOne)

        self.Plabel += Phists
        self.NLabel += Nhists
        self.labels = np.zeros((len(self.Plabel)+len(self.NLabel)))
        self.labels[0:len(self.Plabel)] = 1
        X = np.array(self.Plabel + self.NLabel)
        Y = self.labels
        self.clf.fit(X, Y)
    
    def predict(self, img):
        # 滑动窗口计算出图片中所有小区域的hist
        rate = 2.0376
        img_width = img.shape[1]
        img_height = img.shape[0]
        
        loc = []
        hists = []
        for i in range(8):
            cur_w = int(self.tarW * rate)
            cur_h = int(self.tarH * rate)
            for j in range(0, int(2*img_width/cur_w)):
                for k in range(0, int(2*img_height/cur_h)):
                    leftx = int(cur_w*j/2)
                    lefty = int(cur_h*k/2)
                    NRect = img[lefty:lefty+cur_h, leftx:leftx+cur_w]
                    histOne = np.bincount(NRect.ravel(), minlength=256)
                    # cv2.rectangle(img, (leftx, lefty), (leftx+cur_w, lefty+cur_h), (255, 255, 255), 1)
                    histOne = histOne/np.sum(histOne)
                    hists.append(histOne)
                    loc.append((leftx, lefty, cur_w, cur_h))
            rate = rate / 1.2
            if cur_w*rate < 30 or cur_h*rate < 30:
                break 
        X = np.array(hists)
        Y = self.clf.predict(X)
        x,y,w,h = loc[np.argmax(Y)]
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)
        return Y, loc

    def refreshClassifier(self, PHists, NHists):
        # 刷新分类器
        pass
                
        

def runApp():
    tool = OTBTool()
    imglist, ground_truth = tool.getOneVideo(0)
    img = cv2.imread(imglist[0], 0)
    detector = Detector()
    detector.initDetector(img, ground_truth[0])

    for img2 in imglist:
        frame = cv2.imread(img2, 0)
        Y, loc = detector.predict(frame)
        cv2.imshow('frame', frame)
        cv2.waitKey(10)

if __name__ == '__main__':
    runApp()