import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy import spatial
import copy

#UNCLASSIFIED初始标签， NOISE噪点
UNCLASSIFIED = 0
NOISE = -1
# 曲率搜索半径是3，法向量的搜索半径是3

def pca_compute(data, sort=True):
    """
     SVD分解计算点云的特征值与特征向量
    :param data: 输入数据
    :param sort: 是否将特征值特征向量进行排序
    :return: 特征值与特征向量
    """
    average_data = np.mean(data, axis=0)  # 求均值
    decentration_matrix = data - average_data  # 去中心化
    H = np.dot(decentration_matrix.T, decentration_matrix)  # 求解协方差矩阵H
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)  # SVD求解特征值、特征向量

    if sort:
        sort = eigenvalues.argsort()[::-1]  # 降序排列
        eigenvalues = eigenvalues[sort]  # 索引

    return eigenvalues


def caculate_surface_curvature(cloud, radius=0.003):
    """
    计算点云的表面曲率
    :param cloud: 输入点云
    :param radius: k近邻搜索的半径，默认值为：3m
    :return: 点云中每个点的表面曲率
    """
    points = np.asarray(cloud.points)
    kdtree = o3d.geometry.KDTreeFlann(cloud)
    num_points = len(cloud.points)
    curvature = []  # 储存表面曲率
    for i in range(num_points):
        k, idx, _ = kdtree.search_radius_vector_3d(cloud.points[i], radius)

        neighbors = points[idx, :]
        w = pca_compute(neighbors)  # w为特征值
        delt = np.divide(w[2], np.sum(w), out=np.zeros_like(w[2]), where=np.sum(w) != 0)
        curvature.append(delt)
    curvature = np.array(curvature, dtype=np.float64)
    return curvature


def caculate_nor(a, b):
    '''计算两个a,b法向量的夹角的余弦绝对值与1的差值'''
    final = 1 - abs((a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / \
                    (pow(a[0] ** 2 + a[1] ** 2 + a[2] ** 2, 0.5) * pow(b[0] ** 2 + b[1] ** 2 + b[2] ** 2, 0.5)))
    return final


# 聚类扩展
# dists ： 所有数据两两之间的距离  N x N
# labs :   所有数据的标签 labs N，
# cluster_id ： 一个簇的标号
# eps ： 密度评估半径
# seeds： 用来进行簇扩展的点
# min_points： 半径内最少的点数
def expand_cluster(tree,datas,labs, cluster_id, seeds, eps,threshold_sur,threshold_nor,search_sur, max_sur):
    i = 0
    while i < len(seeds):
        # 获取一个临近点
        Pn = seeds[i]
        search_sur[Pn] = max_sur
        # 如果该点被标记为NOISE 则重新标记
        if labs[Pn] == NOISE:
            labs[Pn] = cluster_id
        # 如果该点没有被标记过
        elif labs[Pn] == UNCLASSIFIED:
            # 进行标记，并计算它的临近点 new_seeds
            labs[Pn] = cluster_id

            new_seeds = tree.query_ball_point(datas[Pn],eps)
            for j in range(len(new_seeds)):
                # 判断种子点的搜索半径内的点是否满足与种子差值的阈值，满足则加入队列
                error_sur = abs(surface_curvature[new_seeds[j]] - surface_curvature[Pn])  # 表面曲率差值
                error_nor = abs(caculate_nor(Normal_Vector[new_seeds[j]],Normal_Vector[Pn])) # 法向量差值
                if error_nor<=threshold_nor and  error_sur<=threshold_sur:
                    seeds = seeds + [new_seeds[j]]
        i = i + 1

def dbscan(datas, eps, min_points,threshold_sur,threshold_nor,search_sur, max_sur):
    """选择初始点为曲率最小点"""
    # 建立kd树
    tree = spatial.KDTree(data=datas)

    # 将所有点的标签初始化为UNCLASSIFIED
    n_points = datas.shape[0]
    labs = [UNCLASSIFIED] * n_points

    cluster_id = 0
    # 先找到曲率最小的点,直到所有的点都已经打完标签
    for i in range(0,n_points):
        point_id = np.argmin(search_sur)
        min_sur = np.min(search_sur)
        if min_sur == max_sur:
            break
        #计算临近点
        seeds = tree.query_ball_point(datas[point_id],eps)

        # 如果临近点数量过少则标记为 NOISE
        if len(seeds) < min_points:
            labs[point_id] = NOISE
            search_sur[point_id] = max_sur
        else:
            # 否则就开启一轮簇的扩张
            cluster_id = cluster_id + 1
            # 标记当前点
            labs[point_id] = cluster_id
            search_sur[point_id] = max_sur
            expand_cluster(tree,datas, labs, cluster_id, seeds, eps,threshold_sur,threshold_nor,search_sur, max_sur)
    return labs, cluster_id

#####################曲率最小为初始点
# def dbscan(datas, eps, min_points,threshold_sur,threshold_nor,search_sur, max_sur):
#     """不选择初始点"""
#     # 建立kd树
#     tree = spatial.KDTree(data=datas)
#
#     # 将所有点的标签初始化为UNCLASSIFIED
#     n_points = datas.shape[0]
#     labs = [UNCLASSIFIED] * n_points
#
#     cluster_id = 0
#     # 遍历所有点,先找到曲率最小的点
#     for point_id in range(0, n_points):
#         # 如果当前点已经处理过了
#         if not (labs[point_id] == UNCLASSIFIED):
#             continue
#
#         # 没有处理过则计算临近点
#         seeds = tree.query_ball_point(datas[point_id],eps)
#
#         # 如果临近点数量过少则标记为 NOISE
#         if len(seeds) < min_points:
#             labs[point_id] = NOISE
#         else:
#             # 否则就开启一轮簇的扩张
#             cluster_id = cluster_id + 1
#             # 标记当前点
#             labs[point_id] = cluster_id
#             expand_cluster(tree,datas, labs, cluster_id, seeds, eps,threshold_sur,threshold_nor,search_sur, max_sur)
#     return labs, cluster_id

if __name__ == "__main__":
    #参数
    threshold_sur = 0.5# 表面曲率阈值
    threshold_nor = 0.6# 法向量余弦阈
    eps = 5#种子搜索半径
    min_points = 5#最小的种子点
    radius_nor = 50#法向量搜索半径
    radius_sur = 60#表面曲率搜索半径


    # # 公司数据
    # datas = np.loadtxt('../data/newfaultidffaultvolume_faultline_small_cross_simulation.csv', delimiter=',')
    # np_points = datas[:, 0:3]  # 取出x,y,z

    datas = np.loadtxt('../code/myfile1.csv', delimiter=',')
    # 数据正则化
    # datas = StandardScaler().fit_transform(datas)
    np_points = datas

    print(len(datas))
    pcd = o3d.geometry.PointCloud()  # pcd类型的数据
    pcd.points = o3d.utility.Vector3dVector(np_points)

    #计算数据点的法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius_nor, max_nn=30))
    Normal_Vector = np.array(pcd.normals)
    #得到数据点的表面曲率
    surface_curvature = np.array(caculate_surface_curvature(pcd, radius_sur)).reshape(len(np_points[:,0]), 1)
    search_sur = copy.deepcopy(surface_curvature)#曲率最小筛选库
    max_sur = np.max(copy.deepcopy(surface_curvature)) + 1  # 最大表面曲率加一




    #代入函数计算
    labs, cluster_id = dbscan(np_points, eps, min_points,threshold_sur,threshold_nor,search_sur, max_sur)
    print("labs of my dbscan")

    #输出各类个数
    print(labs)
    labs = np.array(labs)
    from collections import Counter
    d = Counter(labs)
    d_s = sorted(d.items(), key=lambda x: x[1], reverse=True)
    print(d_s)

    #可视化
    db_labels = labs
    db_max_label = labs.max()
    print(f"point cloud has {db_max_label} clusters")
    # --------------------可视化聚类结果----------------------
    colors = plt.get_cmap("tab20")(db_labels / (db_max_label if db_max_label > 0 else 1))
    colors[db_labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

