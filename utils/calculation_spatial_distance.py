import numpy as np

# ==========================================
# 核心数学工具函数
# ==========================================

def qvec2rotmat(qvec):
    """
    如果你读取的数据是四元数 (qw, qx, qy, qz)，
    可以使用此函数将其转换为旋转矩阵 R。
    """
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def pixel_to_ray(u, v, K, R, T):
    """
    将像素坐标转换为世界坐标系下的射线起点 (Origin) 和方向 (Direction)。
    :param u, v: 像素坐标
    :param K: 内参矩阵 (3x3)
    :param R: 旋转矩阵 (3x3, World-to-Camera)
    :param T: 平移向量 (3,)
    :return: (origin, direction_world)
    """
    # 计算相机中心 (射线起点) 在世界坐标系的位置
    # 公式: center = -R^T * T
    R_inv = R.T
    origin = -R_inv @ T

    # 像素坐标齐次化
    pixel_h = np.array([u, v, 1.0])

    # 转换到相机坐标系方向
    K_inv = np.linalg.inv(K)
    direction_cam = K_inv @ pixel_h

    # 转换到世界坐标系方向
    direction_world = R_inv @ direction_cam
    
    # 归一化
    direction_world = direction_world / np.linalg.norm(direction_world)
    
    return origin, direction_world

def compute_skew_line_distance(O1, D1, O2, D2):
    """
    计算两条异面直线 (Skew Lines) 之间的最短距离。
    :param O1, D1: 射线1的起点和方向
    :param O2, D2: 射线2的起点和方向
    :return: 距离 (float)
    """
    W0 = O1 - O2
    cross_prod = np.cross(D1, D2)
    cross_norm = np.linalg.norm(cross_prod)
    
    # 如果方向向量平行 (叉积接近0)
    if cross_norm < 1e-6:
        return np.linalg.norm(np.cross(D1, W0))
    
    return np.abs(np.dot(W0, cross_prod)) / cross_norm

# ==========================================
# 主计算接口
# ==========================================

def calculate_spatial_distance(pt1, K1, R1, T1, pt2, K2, R2, T2):
    """
    计算两个视角下对应点生成的射线在三维空间中的最短距离。
    
    :param pt1: 视角1的像素坐标 (u, v) / list / tuple / np.array
    :param K1:  视角1的内参矩阵 (3x3)
    :param R1:  视角1的旋转矩阵 (3x3)
    :param T1:  视角1的平移向量 (3,)
    :param pt2: 视角2的像素坐标 (u, v)
    :param K2:  视角2的内参矩阵 (3x3)
    :param R2:  视角2的旋转矩阵 (3x3)
    :param T2:  视角2的平移向量 (3,)
    :return:    空间距离 (float)
    """
    # 1. 计算第一条射线
    O1, D1 = pixel_to_ray(pt1[0], pt1[1], K1, R1, T1)
    
    # 2. 计算第二条射线
    O2, D2 = pixel_to_ray(pt2[0], pt2[1], K2, R2, T2)
    
    # 3. 计算异面直线距离
    dist = compute_skew_line_distance(O1, D1, O2, D2)
    
    return dist

# ==========================================
# 使用示例 (测试用)
# ==========================================
if __name__ == "__main__":
    # 假设你已经从自己的代码中读取到了以下数据：
    
    # 模拟数据：视角 1
    point1 = (1843.5, 940.2)
    K1 = np.array([[2000, 0, 960], [0, 2000, 540], [0, 0, 1]]) # 假设内参
    R1 = np.eye(3)                                          # 假设旋转
    T1 = np.zeros(3)                                        # 假设平移
    
    # 模拟数据：视角 2 (假设向右移动了 1个单位)
    point2 = (1843.5, 940.2) 
    K2 = np.array([[2000, 0, 960], [0, 2000, 540], [0, 0, 1]])
    R2 = np.eye(3)
    T2 = np.array([-1.0, 0, 0]) 

    # 调用函数计算
    distance = calculate_spatial_distance(point1, K1, R1, T1, point2, K2, R2, T2)
    
    print(f"Calculated Spatial Distance: {distance}")