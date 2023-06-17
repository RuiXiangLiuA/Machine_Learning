# 编译时间：2022/8/9 18:05
import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math
import random


def strlist2num(dl):
    # 将字符串列表转化为浮点型列表
    data = []
    for i in range(len(dl)):
        if dl[i] == 'nan' or dl[i] == 'NaN':
            raise ValueError('data is nan')
        data.append(float(dl[i]))
    return np.array(data)


def read_txt(path, row_skip=0, split_char=',', num_range=None, verbose=False):
    """
    read txt file into a np.ndarray.

    Input：
    ------
    path: file path
    row_skip: skip the first rows to read data
    split_char: spliting character
    num_range: data range of each number
    Output：
    ------
    data: data read. data is np.array([]) when reading error happened
                     data is np.array([]) when nan or NaN appears
                     data is np.array([]) when any number is out of range
    """

    try:
        f = open(path, 'r', encoding='utf-8')
        line_list = f.readlines()
        read_lines_num = len(line_list)

        for i in range(read_lines_num):
            line_list[i] = line_list[i].rstrip()

        i = row_skip  # 从第三行开始读取
        data = []
        while i <= read_lines_num - 1:
            data_str = line_list[i].split(split_char)
            data.append(strlist2num(data_str))
            i = i + 1
        f.close()
    except:
        if verbose:
            print("type data of [{}] is wrong...".format(path))
        data = np.array([])
        f.close()
    data = np.array(data)
    if num_range is not None:
        if np.any(data < num_range[0]) or np.any(data > num_range[1]):
            data = np.array([])
            if verbose:
                print("data of [{}] is out of range...".format(path))
    return data


def SVD(points):
    # 二维，三维均适用
    # 二维直线，三维平面
    pts = points.copy()
    # 奇异值分解
    c = np.mean(pts, axis=0)
    A = pts - c  # shift the points
    A = A.T  # 3*n
    u, s, vh = np.linalg.svd(A, full_matrices=False, compute_uv=True)  # A=u*s*vh
    normal = u[:, -1]

    # 法向量归一化
    nlen = np.sqrt(np.dot(normal, normal))
    normal = normal / nlen
    # normal 是主方向的方向向量 与PCA最小特征值对应的特征向量是垂直关系
    # u 每一列是一个方向
    # s 是对应的特征值
    # c >>> 点的中心
    # normal >>> 拟合的方向向量
    return u, s, c, normal


class plane_model(object):
    def __init__(self):
        self.parameters = None

    def calc_inliers(self, points, dst_threshold):
        c = self.parameters[0:3]
        n = self.parameters[3:6]
        dst = abs(np.dot(points - c, n))
        ind = dst < dst_threshold
        return ind

    def estimate_parameters(self, pts):
        num = pts.shape[0]
        if num == 3:
            c = np.mean(pts, axis=0)
            l1 = pts[1] - pts[0]
            l2 = pts[2] - pts[0]
            n = np.cross(l1, l2)
            scale = [n[i] ** 2 for i in range(n.shape[0])]
            # print(scale)
            n = n / np.sqrt(np.sum(scale))
        else:
            _, _, c, n = SVD(pts)

        params = np.hstack((c.reshape(1, -1), n.reshape(1, -1)))[0, :]
        self.parameters = params
        return params

    def set_parameters(self, parameters):
        self.parameters = parameters


def ransac_planefit(points, ransac_n, max_dst, max_trials=1000, stop_inliers_ratio=1.0, initial_inliers=None):
    # RANSAC 平面拟合
    pts = np.array(points.copy())
    num = pts.shape[0]
    cc = np.mean(pts, axis=0)
    iter_max = max_trials
    best_inliers_ratio = 0  # 符合拟合模型的数据的比例
    best_plane_params = None
    best_inliers = None
    best_remains = None
    for i in range(iter_max):
        sample_index = random.sample(range(num), ransac_n)
        sample_points = pts[sample_index, :]
        plane = plane_model()
        plane_params = plane.estimate_parameters(sample_points)
        #  计算内点 外点
        index = plane.calc_inliers(points, max_dst)
        inliers_ratio = pts[index].shape[0] / num

        if inliers_ratio > best_inliers_ratio:
            best_inliers_ratio = inliers_ratio
            best_plane_params = plane_params
            bset_inliers = pts[index]
            bset_remains = pts[index == False]

        if best_inliers_ratio > stop_inliers_ratio:
            # 检查是否达到最大的比例
            print("iter: %d\n" % i)
            print("best_inliers_ratio: %f\n" % best_inliers_ratio)
            break

    return best_plane_params, bset_inliers, bset_remains


def ransac_plane_detection(points, ransac_n, max_dst, max_trials=1000, stop_inliers_ratio=1.0, initial_inliers=None,
                           out_layer_inliers_threshold=230, out_layer_remains_threshold=230):
    inliers_num = out_layer_inliers_threshold + 1
    remains_num = out_layer_remains_threshold + 1

    plane_set = []
    plane_inliers_set = []
    plane_inliers_num_set = []

    data_remains = np.copy(points)

    i = 0

    while inliers_num > out_layer_inliers_threshold and remains_num > out_layer_remains_threshold:
        # robustly fit line only using inlier data with RANSAC algorithm
        best_plane_params, pts_inliers, pts_outliers = ransac_planefit(data_remains, ransac_n, max_dst,
                                                                       max_trials=max_trials,
                                                                       stop_inliers_ratio=stop_inliers_ratio)

        inliers_num = pts_inliers.shape[0]
        remains_num = pts_outliers.shape[0]

        if inliers_num > out_layer_inliers_threshold:
            plane_set.append(best_plane_params)
            plane_inliers_set.append(pts_inliers)
            plane_inliers_num_set.append(inliers_num)
            i = i + 1
            print('------------> %d <--------------' % i)
            print(best_plane_params)

        data_remains = pts_outliers

    # sorting
    plane_set = [x for _, x in sorted(zip(plane_inliers_num_set, plane_set), key=lambda s: s[0], reverse=True)]
    plane_inliers_set = [x for _, x in
                         sorted(zip(plane_inliers_num_set, plane_inliers_set), key=lambda s: s[0], reverse=True)]

    return plane_set, plane_inliers_set, data_remains


def show_3dpoints(pointcluster, s=None, colors=None, quiver=None, q_length=10, tri_face_index=None):
    # pointcluster should be a list of numpy ndarray
    # This functions would show a list of pint cloud in different colors
    n = len(pointcluster)
    if colors is None:
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'tomato', 'gold']
        if n < 10:
            colors = np.array(colors[0:n])
        else:
            colors = np.random.rand(n, 3)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if s is None:
        s = np.ones(n) * 2

    for i in range(n):
        ax.scatter(pointcluster[i][:, 0], pointcluster[i][:, 1], pointcluster[i][:, 2], s=s[i], c=[colors[i]],
                   alpha=0.6)

    if not (quiver is None):
        c1 = [random.random() for _ in range(len(quiver))]
        c2 = [random.random() for _ in range(len(quiver))]
        c3 = [random.random() for _ in range(len(quiver))]
        c = []
        for i in range(len(quiver)):
            c.append((c1[i], c2[i], c3[i]))
        cp = []
        for i in range(len(quiver)):
            cp.append(c[i])
            cp.append(c[i])
        c = c + cp
        ax.quiver(quiver[:, 0], quiver[:, 1], quiver[:, 2], quiver[:, 3], quiver[:, 4], quiver[:, 5], length=q_length,
                  arrow_length_ratio=.2, pivot='tail', normalize=False, color=c)

    if not (tri_face_index is None):
        for i in range(len(tri_face_index)):
            for j in range(tri_face_index[i].shape[0]):
                index = tri_face_index[i][j].tolist()
                index = index + [index[0]]
                ax.plot(*zip(*pointcluster[i][index]))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # ax.set_ylim([-20,60])

    plt.show()

    return 0


if __name__ == "__main__":

    #自己的数据
    datas = np.loadtxt('../data/myfile.csv', delimiter=',')
    # 公司数据
    # data = np.loadtxt('../data/newfaultidffaultvolume_faultline_small_cross_simulation.csv', delimiter=',')
    # datas = data[:, 0:3]  # 取出x,y,z
    #

#选取ransac_n个随机点组成平面，max_dst判断该点距离平面的阈值
    plane_set, plane_inliers_set, data_remains = ransac_plane_detection(  datas ,ransac_n=3, max_dst=5.5, max_trials=1000,
                                                                        stop_inliers_ratio=1.0, initial_inliers=None,
                                                                        out_layer_inliers_threshold=230,
                                                                        out_layer_remains_threshold=230)
    plane_set = np.array(plane_set)

    print("================= 平面参数 ====================")
    print(plane_set)
    # 绘图
    show_3dpoints(plane_inliers_set)
    print("over!!!")
