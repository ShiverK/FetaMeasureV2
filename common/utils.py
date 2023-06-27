import SimpleITK as sitk
import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage import morphology
from sklearn.linear_model import LinearRegression
import math


def get_cntr_form_contour(contour):
    """
    利用轮廓的图像矩计算中点
    :param contours:
    :return:
    """
    m1 = cv2.moments(contour)
    x1 = int(m1["m10"] / m1["m00"])
    y1 = int(m1["m01"] / m1["m00"])
    return (x1, y1)


def get_cntr_form_slice(msk_slice):
    """
    利用切片的二值化图像计算中点
    :param msk_slice: 二值化切片
    :return:
    """
    contours, hierarchy = cv2.findContours(msk_slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    m1 = cv2.moments(contours[0])
    x1 = math.ceil(m1["m10"] / m1["m00"])
    y1 = math.ceil(m1["m01"] / m1["m00"])
    return [x1, y1]


def euclidean_distance_point2line(k, h, pointIndex: tuple):
    """
    计算一个点到某条直线的euclidean distance
    :param k: 直线的斜率，float类型
    :param h: 直线的截距，float类型
    :param pointIndex: 一个点的坐标，（横坐标，纵坐标），tuple类型
    :return: 点到直线的euclidean distance，float类型
    """
    x = pointIndex[0]
    y = pointIndex[1]
    theDistance = math.fabs(h + k * (x - 0) - y) / (math.sqrt(k * k + 1))
    return theDistance


def rotate_point_from_cntr(point1, cntr, angle, height):
    """
    点point1绕点point2旋转angle后的点
    ======================================
    在平面坐标上，任意点P(x1,y1)，绕一个坐标点Q(x2,y2)旋转θ角度后,新的坐标设为(x, y)的计算公式：
    x= (x1 - x2)*cos(θ) - (y1 - y2)*sin(θ) + x2 ;
    y= (x1 - x2)*sin(θ) + (y1 - y2)*cos(θ) + y2 ;
    ======================================
    将图像坐标(x,y)转换到平面坐标(x`,y`)：
    x`=x
    y`=height-y
    :param point1:
    :param cntr: base point (基点)
    :param angle: 旋转角度，正：表示逆时针，负：表示顺时针
    :param height:
    :return:
    """
    x1, y1 = point1
    x2, y2 = cntr
    # 将图像坐标转换到平面坐标
    y1 = height - y1
    y2 = height - y2
    x = (x1 - x2) * np.cos(np.pi / 180.0 * angle) - (y1 - y2) * np.sin(np.pi / 180.0 * angle) + x2
    y = (x1 - x2) * np.sin(np.pi / 180.0 * angle) + (y1 - y2) * np.cos(np.pi / 180.0 * angle) + y2
    # 将平面坐标转换到图像坐标
    y = height - y
    return [x.astype(int), y.astype(int)]


def rotate_angle(img, angle, cntr):
    """
    rotate img using rect
    :param img: img needed to rotate
    :param rect: the minAreaRect
    :return:
    """
    angle = -angle
    rows, cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D(cntr, angle, 1)
    img_rot = cv2.warpAffine(img, M, (cols, rows))
    return img_rot

