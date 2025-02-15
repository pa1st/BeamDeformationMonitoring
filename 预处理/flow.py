"""
-*- coding: utf-8 -*-
@Time    : 2025/1/13 20:31
@Author  : PA1ST
@File    : flow.py
@Software: PyCharm
"""
import open3d as o3d
import numpy as np


def preprocess_point_cloud(pcd_path, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0, normalize=False):
    """
    使用 Open3D 对点云进行预处理。

    参数:
    - pcd_path: str，点云文件的路径
    - voxel_size: float，用于下采样的体素大小
    - nb_neighbors: int，统计滤波中用于计算平均距离的邻域点数量
    - std_ratio: float，统计滤波中用于剔除噪声点的标准差倍数
    - normalize: bool，是否对预处理后的点云进行归一化

    返回:
    - pcd: 预处理完成后的 open3d.geometry.PointCloud 对象
    """

    # 1. 加载点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    print("加载点云完成。点云共有 {} 个点.".format(len(pcd.points)))

    # 2. 去噪 (统计滤波法)
    #   nb_neighbors: 邻域点数量
    #   std_ratio:    超过平均距离 std_ratio 倍的点认为是噪声
    pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                    std_ratio=std_ratio)
    print("去噪后，点云留下 {} 个点.".format(len(pcd_clean.points)))

    # 3. 下采样
    pcd_down = pcd_clean.voxel_down_sample(voxel_size=voxel_size)
    print("下采样后，点云共有 {} 个点.".format(len(pcd_down.points)))

    # 4. 估计法线
    #   search_param: 搜索半径与 KNN 的设置可根据点云密度适当调整
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.05, max_nn=30))
    print("法线估计完成。")

    if normalize:
        # 5. 归一化处理
        # 以去噪后的点云为基础，将点云居中并缩放到一定范围
        points = np.asarray(pcd_down.points)
        center = np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(points - center, axis=1))
        # 将点云中心移到原点，并按最大半径做缩放
        normalized_points = (points - center) / max_dist
        pcd_down.points = o3d.utility.Vector3dVector(normalized_points)
        print("完成归一化处理。")

    return pcd_down


if __name__ == "__main__":
    pcd_path = "test1.pcd"

    # 执行点云预处理
    processed_pcd = preprocess_point_cloud(
        pcd_path,
        voxel_size=0.04,
        nb_neighbors=20,
        std_ratio=2,
        normalize=True
    )
    o3d.visualization.draw_geometries([processed_pcd])

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(processed_pcd, voxel_size=0.005)
    o3d.visualization.draw_geometries([voxel_grid])

    octree = o3d.geometry.Octree(max_depth=7)
    octree.create_from_voxel_grid(voxel_grid)
    o3d.visualization.draw_geometries([octree])