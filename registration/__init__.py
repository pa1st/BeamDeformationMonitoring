"""
-*- coding: utf-8 -*-
@Time    : 2025/1/14 17:16
@Author  : PA1ST
@File    : __init__.py.py
@Software: PyCharm
"""
import open3d as o3d
import numpy as np

# 加载点云数据
def load_point_clouds(file1, file2):
    pcd1 = o3d.io.read_point_cloud(file1)
    pcd2 = o3d.io.read_point_cloud(file2)
    return pcd1, pcd2

# 计算 FPFH 特征
def compute_fpfh_feature(pcd, voxel_size):
    radius_normal = voxel_size * 2
    radius_feature = voxel_size * 5

    # 估计法线
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # 计算 FPFH 特征
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh

# 粗配准 (FPFH + RANSAC)
def coarse_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result

# 精配准 (ICP)
def refine_registration(source, target, transformation, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result

# 主流程
def main():
    # 加载点云文件
    file1 = "bridge1.pcd"
    file2 = "bridge2.pcd"
    source, target = load_point_clouds(file1, file2)

    # 设置体素大小
    voxel_size = 0.05
    
    # 计算 FPFH 特征
    source_fpfh = compute_fpfh_feature(source, voxel_size)
    target_fpfh = compute_fpfh_feature(target, voxel_size)

    # 粗配准
    coarse_result = coarse_registration(source, target, source_fpfh, target_fpfh, voxel_size)
    print("粗配准结果：")
    print(coarse_result)

    # 精配准
    refined_result = refine_registration(source, target, coarse_result.transformation, voxel_size)
    print("精配准结果：")
    print(refined_result)

    # 可视化结果
    source.transform(refined_result.transformation)
    o3d.visualization.draw_geometries([source, target], window_name="点云配准结果")

if __name__ == "__main__":
    main()
