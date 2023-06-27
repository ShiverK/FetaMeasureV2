from .axial_stand import AxialStand
import cv2
import numpy as np
from skimage.morphology import skeletonize




class AxialMeter(AxialStand):
    def __init__(self, image_path, mask_path, stand_axcodes=str, label_info=None):
        super(AxialMeter, self).__init__(image_path, mask_path, stand_axcodes)
        self.image_path = image_path
        self.mask_path = mask_path
        self.stand_axcodes = stand_axcodes
        # label_info  {"HC": 5, 'BA': 4, 'CSP': 1, 'AD': 3, 'TCD': 2}
        self.label_info = label_info
        self.stand = AxialStand(self.image_path, self.mask_path, self.stand_axcodes, self.label_info)


    def run(self):
        axial_stand = AxialStand(self.image_path, self.mask_path, self.stand_axcodes)
        img_stand, msk_stand = axial_stand.run_standlized()

    def run_2Dstand_2Dmeasure(self, msk_array, label_name):
        """
        基于2D图像骨骼提取的标准化和测量
        :param msk_array:
        :param label_name:
        :return:
        """
        label = self.label_info[label_name]
        msk_array_label = np.where(msk_array == label, 1, 0)
        msk_max_id = np.argmax(np.sum(msk_array_label, axis=(0, 1)))
        return msk_array_label[:, :, msk_max_id].astype(np.uint8)

    def get_refer_max_slice(self, msk_array, label_name):
        """
        获取标签的最大索引的二值化切片
        :param msk_array:
        :param label_name:
        :return:
        """
        label = self.label_info[label_name]
        msk_array_label = np.where(msk_array == label, 1, 0)
        msk_max_id = np.argmax(np.sum(msk_array_label, axis=(0, 1)))
        return msk_array_label[:, :, msk_max_id].astype(np.uint8)

    def get_refer_max_slice(self, msk_array, label_name):
        """
        获取标签的最大索引的二值化切片
        :param msk_array:
        :param label_name:
        :return:
        """
        label = self.label_info[label_name]
        msk_array_label = np.where(msk_array == label, 1, 0)
        msk_max_id = np.argmax(np.sum(msk_array_label, axis=(0, 1)))
        return msk_array_label[:, :, msk_max_id].astype(np.uint8)

    def get_refer_max_slice(self, msk_array, label_name):
        """
        获取标签的最大索引的二值化切片
        :param msk_array:
        :param label_name:
        :return:
        """
        label = self.label_info[label_name]
        msk_array_label = np.where(msk_array == label, 1, 0)
        msk_max_id
