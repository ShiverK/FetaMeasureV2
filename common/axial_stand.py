# 用于获取轴位旋转角度和标准化图像
import SimpleITK as sitk
import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import binary_fill_holes, binary_erosion
from skimage import morphology
from sklearn.linear_model import LinearRegression
import math
from .utils import get_cntr, euclidean_distance, get_cntr_form_slice, rotate_point, rotate_angle
from pathlib import Path


class AxialStand(object):
    def __init__(self, image_path: Path, mask_path: Path, stand_axcodes: str, label_info=None):
        super(AxialStand, self).__init__()
        self.n_slices = None
        if label_info is None:
            label_info = {"HC": 5, 'BA': 4, 'CSP': 1, 'AD': 3, 'TCD': 2}
        self.axcodes = stand_axcodes
        assert self.axcodes in ['LPI', 'RPI', 'LPS', 'RPS', 'LAS', 'RAS', 'LAI',
                                'RAI'], 'axcodes must be LPI, RPI, LPS, RPS, LAS, RAS, LAI, RAI'

        self.image_path = image_path
        self.mask_path = mask_path
        self.image_name = image_path.name.split('.')[0]

        assert os.path.exists(self.image_path), 'image path is not exists'
        assert os.path.exists(self.mask_path), 'mask path is not exists'

        self.label_info = label_info
        self.stand_info = dict()
        self.img_info = dict()
        self.msk_info = dict()
        self.stand_angle = 0

    def load_data(self, image_path, mask_path, axcodes):
        """
        使用SimpleITK读取图像和标签，并将图像和标签转换为numpy数组
        :param image_path:
        :param mask_path:
        :param axcodes:
        :return:
        """
        img_org = sitk.ReadImage(image_path)
        msk_org = sitk.ReadImage(mask_path)

        img = sitk.DICOMOrient(img_org, axcodes)
        msk = sitk.DICOMOrient(msk_org, axcodes)

        self.img_info = {'size': img.GetSize(), 'org_origin': img_org.GetOrigin(), 'origin': img.GetOrigin(),
                         'spacing': img.GetSpacing(),
                         'origin_direction': img_org.GetDirection(), 'trans_direction': img.GetDirection()}
        self.msk_info = {'size': msk.GetSize(), 'org_origin': msk_org.GetOrigin(), 'origin': msk.GetOrigin(),
                         'spacing': msk.GetSpacing(),
                         'origin_direction': msk_org.GetDirection(), 'trans_direction': msk.GetDirection()}
        img_array = sitk.GetArrayFromImage(img).transpose([1, 2, 0]).astype(np.float16)
        img_norm_arr = ((img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * 255).astype(
            np.uint8)

        msk_array = sitk.GetArrayFromImage(msk).transpose([1, 2, 0]).astype(np.uint8)
        return img_norm_arr, msk_array

    def get_refer_max_slice(self, msk_array, label_name):
        """
        获取指定标签面积的最大索引的二值化切片
        :param msk_array:
        :param label_name:
        :return:
        """
        label = self.label_info[label_name]
        msk_array_label = np.where(msk_array == label, 1, 0)
        msk_max_id = np.argmax(np.sum(msk_array_label, axis=(0, 1)))
        return msk_array_label[:, :, msk_max_id].astype(np.uint8), msk_max_id

    def get_fitline_2d_skeletonize(self, msk_array, label_name, DoMorphology=True):
        """
        2023.4.22
        基于2d图像的骨干提取和线性拟合
        :param msk_array:
        :param label_name:
        :return:拟合的直线
        """
        assert label_name in self.label_info.keys(), 'labelname must in label_info'
        liner = LinearRegression()
        msk_slice, _ = self.get_refer_max_slice(msk_array, label_name)
        # 闭运算去噪
        if DoMorphology:
            # 设置卷积核
            kernel = np.ones((3, 3), np.uint8)
            # 进行图像膨胀
            msk_slice = cv2.dilate(msk_slice, kernel, iterations=3)
            # 腐蚀去噪
            msk_slice = cv2.erode(msk_slice, kernel, iterations=1)
        # 骨架提取
        msk_slice_sk = skeletonize(msk_slice)
        row, col = np.nonzero(msk_slice_sk)
        liner.fit(col.reshape(-1, 1), row)
        # 获取线性拟合的斜率和截距
        w = float(liner.coef_[0])
        b = float(liner.intercept_)
        return [w, b]

    def get_masked_slice(self, img_array, msk_array, label_name):
        """
        获取目标标签面积最大的包含原图信息的掩码切片
        :param msk_array:
        :param label_name:
        :return:
        """
        assert label_name in self.label_info.keys(), 'labelname must in label_info'
        assert img_array.shape == msk_array.shape, 'image and mask shape must be same'
        msk_slice, msk_id = self.get_refer_max_slice(msk_array, label_name)
        return img_array[:, :, msk_id] * msk_slice

    def get_sliceID_contains_label(self, msk_array):
        """

        :param msk_array:
        :return:
        """
        refer_arr_hc = np.where(msk_array == self.label_info['HC'], 1, 0)
        refer_arr_csp = np.where(msk_array == self.label_info['CSP'], 1, 0)
        max_area_idx_hc = np.argmax(np.sum(refer_arr_hc, axis=(0, 1)))

        # 判断大脑面积最大的层内是否有透明隔腔标签
        if msk_array[:, :, max_area_idx_hc].__contains__(self.label_info['CSP']):
            '''计算大脑的标签与透明隔腔标签的面积比，滤除过小的透明隔腔标签'''
            hc_area = np.sum(refer_arr_hc[:, :, max_area_idx_hc])
            csp_area = np.sum(refer_arr_csp[:, :, max_area_idx_hc])
            # 计算面积比
            if csp_area / hc_area > 0.05:
                max_area_idx_csp = max_area_idx_hc
            else:
                max_area_idx_csp = np.argmax(np.sum(refer_arr_csp, axis=(0, 1)))
        else:
            max_area_idx_csp = np.argmax(np.sum(refer_arr_csp, axis=(0, 1)))
        # debug
        # print('max_area_idx_hc:', max_area_idx_hc, 'max_area_idx_csp:', max_area_idx_csp)
        return max_area_idx_hc, max_area_idx_csp

    def get_refer_slice(self, msk_array, label_name, sliceID):
        """
        获取指定标签的指定层的切片
        :param msk_array:
        :param label_name:
        :return:
        """
        assert label_name in self.label_info.keys(), 'labelname must in label_info'
        assert msk_array.shape[2] > sliceID >= 0, 'sliceID must less than msk_array.shape[2]'
        label = self.label_info[label_name]
        msk_array_label = np.where(msk_array == label, 1, 0)
        return msk_array_label[:, :, sliceID].astype(np.uint8)

    def get_refer_masked_slice(self, img_array, msk_array, label_name, sliceID):
        """
        获取指定标签的指定层的掩码切片
        :param msk_array:
        :param msk_array:
        :param label_name:
        :param sliceID:
        :return:
        """
        assert label_name in self.label_info.keys(), 'labelname must in label_info'
        assert msk_array.shape[2] > sliceID >= 0, 'sliceID must less than msk_array.shape[2]'
        assert img_array.shape == msk_array.shape, 'image and mask shape must be same'
        label = self.label_info[label_name]
        msk_array_label = np.where(msk_array == label, 1, 0)
        msk_slice = msk_array_label[:, :, sliceID].astype(np.uint8)
        return img_array[:, :, sliceID] * msk_slice

    def rotate_slice(self, msk_slice, angle, cntr):
        """
        旋转切片
        :param msk_slice:
        :param angle:
        :return:
        """
        angle = -angle
        rows, cols = msk_slice.shape[0], msk_slice.shape[1]
        M = cv2.getRotationMatrix2D(cntr, angle, 1)
        img_rot = cv2.warpAffine(msk_slice, M, (cols, rows))
        return img_rot

    def get_angle_ellipse(self, msk_slice, DoMorphology=True):
        """

        :param msk_array:
        :param label_name:
        :param DoMorphology:
        :return:
        """
        # 闭运算去噪
        if DoMorphology:
            # 设置卷积核
            kernel = np.ones((3, 3), np.uint8)
            # 进行图像膨胀
            msk_slice = cv2.dilate(msk_slice, kernel, iterations=3)
            # 腐蚀去噪
            msk_slice = cv2.erode(msk_slice, kernel, iterations=1)
        # 椭圆拟合
        contours, hierarchy = cv2.findContours(msk_slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area_contour = []
        for k in range(len(contours)):
            area_contour.append(cv2.contourArea(contours[k]))
        # use the largest mask
        try:
            max_idx = np.argmax(np.array(area_contour))
        except:
            max_idx = 0
        cntr_ep, axes_ep, angle_ep = cv2.fitEllipseDirect(contours[max_idx])
        ellipse = tuple([cntr_ep, axes_ep, angle_ep])
        cntr_ep = [math.ceil(i) for i in cntr_ep]
        return angle_ep, cntr_ep, ellipse

    def get_angle_2D_skeletonize(self, msk_array):
        """
        2023.4.22
        基于2d图像的骨干提取和线性拟合获取旋转角度
        使用头围、大脑和透明隔腔的标签
        通过比较侧脑室中点与中线的距离差值判断对称度

        2023.4.25
        该方法中，大脑和头围标签骨架提取效果不加
        已弃用
        :param msk_array:
        :return:
        """
        # 获取头围、大脑和透明隔腔的提取骨架的拟合直线
        head_line = self.get_fitline_2d_skeletonize(msk_array, 'HC')
        brain_line = self.get_fitline_2d_skeletonize(msk_array, 'BA')
        csp_line = self.get_fitline_2d_skeletonize(msk_array, 'CSP')
        line_list = [head_line, brain_line, csp_line]

        # 获取左右侧闹的最大面积切片和轮廓
        ad_slice = self.get_refer_max_slice(msk_array, 'AD')
        ad_counters = cv2.findContours(ad_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        assert len(ad_counters) == 2, 'In max area slice of ad, there are more than two contours'
        # 获取左右侧脑室的中点
        ad_cntr0 = get_cntr(ad_counters[0])
        ad_cntr1 = get_cntr(ad_counters[1])

        # 计算拟合的直接与左右侧脑室中点的距离插值，取差值最小的直线
        dist_fitline = [abs(euclidean_distance(head_line[0], head_line[1], ad_cntr0) -
                            euclidean_distance(head_line[0], head_line[1], ad_cntr1)),
                        abs(euclidean_distance(brain_line[0], brain_line[1], ad_cntr0) -
                            euclidean_distance(brain_line[0], brain_line[1], ad_cntr1)),
                        abs(euclidean_distance(csp_line[0], csp_line[1], ad_cntr0) -
                            euclidean_distance(csp_line[0], csp_line[1], ad_cntr1))]

        fit_ids = list.index(dist_fitline, min(dist_fitline))
        fit_line = line_list[fit_ids]

        return fit_line

    def get_angle_2D_multi_ssim(self, img_array, msk_array):
        """
        2023.4.25
        基于2d图像的骨干提取+线性拟合和椭圆拟合获取旋转角度
        使用头围和透明隔腔的标签
        通过SSIM判断对称度
        :param img_array:
        :param msk_array:
        :return:
        """
        # 获取透明隔腔的提取骨架的拟合直线
        csp_fitline = self.get_fitline_2d_skeletonize(msk_array, 'CSP', DoMorphology=True)
        fitline_angle = 90 + np.rad2deg(np.arctan(csp_fitline[0]))

        # 获取头围和透明隔腔的椭圆拟合
        hc_slice_id, csp_slice_id = self.get_sliceID_contains_label(msk_array)
        hc_slice = self.get_refer_slice(msk_array, 'HC', hc_slice_id)
        csp_slice = self.get_refer_slice(msk_array, 'CSP', csp_slice_id)

        angle_csp_ep, cntr_csp_ep, csp_ellipse = self.get_angle_ellipse(csp_slice, DoMorphology=True)
        angle_hc_ep, cntr_hc_ep, hc_ellipse = self.get_angle_ellipse(hc_slice, DoMorphology=True)

        # 获取包含原始图像内容的头围掩码
        head_slice = self.get_refer_masked_slice(img_array, msk_array, 'HC', hc_slice_id)

        # 获取初始旋转的切片
        rot_slice_ep_csp = self.rotate_slice(head_slice, 180 - angle_csp_ep, cntr_csp_ep)
        rot_slice_ep_hc = self.rotate_slice(head_slice, 180 - angle_hc_ep, cntr_hc_ep)
        cntr_fitline = get_cntr_form_slice(hc_slice)
        rot_slice_fitline = self.rotate_slice(head_slice, 180 - fitline_angle, cntr_fitline)

        # 分别计算三种方法的SSIM
        slice_list = [rot_slice_ep_csp, rot_slice_ep_hc, rot_slice_fitline]
        cntr_list = [cntr_csp_ep, cntr_hc_ep, cntr_fitline]
        angle_list = [angle_csp_ep, angle_hc_ep, fitline_angle]

        # cut_img = list()
        # cut_temp = list()
        # cos_list = list()
        ssim_list = list()
        # mr_list = list()

        from sklearn.metrics.pairwise import cosine_similarity
        from skimage.metrics import structural_similarity
        from sklearn import metrics as mr

        for idx, slice in enumerate(slice_list):
            cntr = cntr_list[idx]
            widlist = slice[cntr[1], :]
            width_half = max([abs(cntr[0] - np.min(np.nonzero(widlist))), abs(cntr[0] - np.max(np.nonzero(widlist)))])
            heilist = slice[:, cntr[0]]
            height_half = max([abs(cntr[1] - np.min(np.nonzero(heilist))), abs(cntr[1] - np.max(np.nonzero(heilist)))])

            # 头围中心切片
            cut = slice[cntr[1] - height_half:cntr[1] + height_half, cntr[0] - width_half:cntr[0] + width_half]

            cutleft = cut[:, 0:math.ceil(cut.shape[1] / 2)]
            cuttemp = cut[:, math.ceil(cut.shape[1] / 2):cut.shape[1]]
            cutright = np.flip(cuttemp, axis=1)

            # cut_temp.append(cutleft)
            # cut_temp.append(cutright)
            # cut_img.append(cut_temp)
            # cut_temp = list()

            ssim = structural_similarity(cutleft, cutright)
            ssim_list.append(ssim)

            # mutual_infor = mr.mutual_info_score(np.reshape(cutleft, -1), np.reshape(cutright, -1))
            # mr_list.append(mutual_infor)

        # 获取SSIM结果最高的索引
        best_ids = ssim_list.index(max(ssim_list))
        self.stand_info = {'methods': ['CSP_skeletonize', 'HC_ellipse', 'CSP_ellipse'], 'ssim': ssim_list,
                           'angle_list': angle_list, 'best_ids': best_ids, 'cntr_list': cntr_list,
                           'ellipses': [hc_ellipse, csp_ellipse], 'fitline': csp_fitline,
                           'rotate_cntr': cntr_list[best_ids]}

        return angle_list[best_ids]

    def run_standlized(self):
        """
        2023.4.25
        基于2d图像的骨干提取+线性拟合和椭圆拟合获取旋转角度
        基于SSIM计算对称性
        :return:
        """

        img_norm_arr, msk_array = self.load_data(self.image_path, self.mask_path, self.axcodes)
        angle = self.get_angle_2D_multi_ssim(img_norm_arr, msk_array)
        self.n_slices = img_norm_arr.shape[2]
        # 根据小脑和透明隔腔的标签判断是否翻转
        TCD_slice, _ = self.get_refer_max_slice(msk_array, 'TCD')
        TCD_cntr = get_cntr_form_slice(TCD_slice)
        CSP_cntr = self.stand_info['cntr_list'][2]

        # debug: 头围面积最大的标签，用于检验
        _, HC_id = self.get_refer_max_slice(msk_array, 'HC')

        rotate_cntr = self.stand_info['rotate_cntr']

        TCD_cntr_trans = rotate_point(TCD_cntr, rotate_cntr, angle, img_norm_arr.shape[1])
        CSP_cntr_trans = rotate_point(CSP_cntr, rotate_cntr, angle, img_norm_arr.shape[1])

        if CSP_cntr_trans[1] > TCD_cntr_trans[1]:
            angle_trans = 180 - angle
        else:
            angle_trans = -angle

        self.stand_angle = angle_trans

        img_std = np.zeros_like(img_norm_arr, dtype=np.uint8)
        msk_std = np.zeros_like(msk_array, dtype=np.uint8)

        for idx in range(self.n_slices):
            img_slice_norm = img_norm_arr[:, :, idx]
            msk_slice = msk_array[:, :, idx]

            img_slice_rotated = self.rotate_slice(img_slice_norm, angle_trans, rotate_cntr)
            classes = list(np.unique(msk_slice))
            msk_slice_rotated = np.zeros_like(msk_slice, dtype=np.uint8)

            for c in classes:
                n_mask_slice = np.where(msk_slice == c, 1, 0)
                n_mask_slice = n_mask_slice.astype(np.uint8)
                n_mask_slice = self.rotate_slice(n_mask_slice, angle_trans, rotate_cntr)
                msk_slice_rotated[n_mask_slice != 0] = c

            img_std[:, :, idx] = img_slice_rotated
            msk_std[:, :, idx] = msk_slice_rotated

        stand_info_total = {'img_info': self.img_info, 'msk_info': self.msk_info, 'stand_info': self.stand_info,
                            'label_info': self.label_info, 'test_id': HC_id, 'angle_trans': angle_trans,
                            'check_points': [TCD_cntr, CSP_cntr], 'check_points_trans': [TCD_cntr_trans, CSP_cntr_trans]}

        return img_std, msk_std, stand_info_total
