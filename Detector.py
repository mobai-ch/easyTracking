'''
再这个文件中，我们尝试去用滑动窗口（传统艺能）实现一个检测器，
毕竟这次就是做一些比较复古的事情，这个和TLD不太一样, 我是仅仅使用了直方图+滑动窗口+SVM。
'''

import cv2
import numpy as np
from sklearn.svm import SVC


