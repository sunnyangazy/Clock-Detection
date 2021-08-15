import cv2
import numpy as np
import sys

"""
针对油气站场压力表目标识别
利用霍夫圆环检测
By SlowNorth 21-8-14
"""

def findGuage(path):
    img1 = cv2.imread(path)
    try:
        if img1 is None:
            raise Exception('找不到图片! 图片路径有误。')
    except Exception as err:
        print(err)
        sys.exit(1)
    img1shape = img1.shape
    img1 = cv2.resize(img1, (int(img1shape[1]/2), int(img1shape[0]/2)))

    canny = cv2.cvtColor(cv2.GaussianBlur(img1, (7, 7), 0), cv2.COLOR_BGR2GRAY)

    '''
    找到仪表区域
    '''
    circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50, minRadius=50)
    circles = np.uint16(np.around(circles))
    img1_cp = img1.copy()
    for i in circles[0, 0:3]:
        # 画圆圈
        cv2.circle(img1_cp, (i[0], i[1]), 2, (0, 0, 255), 3)

    _circles = circles[0,:3] #投票数最大的圆
    _circles = sorted(_circles,key=lambda x:x[2],reverse=True)
    the_circle = _circles[0]
    cv2.circle(img1_cp, (the_circle[0], the_circle[1]), the_circle[2], (0, 0, 255), 3)
    cv2.imwrite('j_circles.jpg', img1_cp)

if __name__ == '__main__':
    path = 'images/j.jpg'
    findGuage(path)