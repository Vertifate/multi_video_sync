import numpy as np

def calculate_batch_spatial_distances(pts1, K1, R1, T1, pts2, K2, R2, T2):
    """
    批量计算两组匹配点对应的空间射线距离。
    
    :param pts1: 视角1的点集，形状 (N, 2)，格式 [(u1, v1), (u2, v2), ...]
    :param K1:   视角1的内参矩阵 (3x3)
    :param R1:   视角1的旋转矩阵 (3x3, World-to-Camera)
    :param T1:   视角1的平移向量 (3,)
    :param pts2: 视角2的点集，形状 (N, 2)
    :param K2:   视角2的内参矩阵 (3x3)
    :param R2:   视角2的旋转矩阵 (3x3)
    :param T2:   视角2的平移向量 (3,)
    :return:     距离数组，形状 (N,)
    """
    # 确保输入是 numpy 数组
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    
    if pts1.shape[0] != pts2.shape[0]:
        raise ValueError(f"两个点集的数量不一致: {pts1.shape[0]} vs {pts2.shape[0]}")

    # ==========================================
    # 1. 批量计算射线 (Vectorized Pixel to Ray)
    # ==========================================
    def batch_pixel_to_ray(pts, K, R, T):
        # 准备数据
        N = pts.shape[0]
        R_inv = R.T
        K_inv = np.linalg.inv(K)
        
        # 1. 计算原点 (所有射线共享同一个相机中心)
        # Origin = -R^T * T
        origin = -R_inv @ T  # shape (3,)

        # 2. 像素坐标齐次化 (N, 2) -> (3, N)
        # 添加一列 1.0
        ones = np.ones((N, 1))
        pixel_h = np.hstack([pts, ones]).T  # 变为 (3, N) 以便矩阵乘法
        
        # 3. 转换方向 (Camera系)
        dir_cam = K_inv @ pixel_h       # (3, 3) @ (3, N) -> (3, N)
        
        # 4. 转换方向 (World系)
        dir_world = R_inv @ dir_cam     # (3, 3) @ (3, N) -> (3, N)
        
        # 5. 归一化方向向量 (按列归一化)
        norms = np.linalg.norm(dir_world, axis=0)
        dir_world_norm = dir_world / norms
        
        return origin, dir_world_norm.T # 返回 (3,) 和 (N, 3)

    # 获取两组射线的参数
    O1, D1 = batch_pixel_to_ray(pts1, K1, R1, T1) # O1: (3,), D1: (N, 3)
    O2, D2 = batch_pixel_to_ray(pts2, K2, R2, T2) # O2: (3,), D2: (N, 3)

    # ==========================================
    # 2. 批量计算异面直线距离
    # ==========================================
    
    # 两个相机中心之间的向量 W0 = O1 - O2
    W0 = O1 - O2 # shape (3,)
    
    # 计算所有对应射线的叉积 (D1 x D2)
    # np.cross 在 axis=1 上操作，结果 shape (N, 3)
    cross_prod = np.cross(D1, D2) 
    
    # 计算叉积的模长 (分母)
    cross_norm = np.linalg.norm(cross_prod, axis=1) # shape (N,)
    
    # 计算分子: |W0 · (D1 x D2)|
    # 由于 W0 是常数向量，我们将其广播到 N 个点上进行点积
    # dot(A, B) row-wise 可以写成 sum(A * B, axis=1)
    numerator = np.abs(np.sum(W0 * cross_prod, axis=1)) # shape (N,)
    
    # 处理分母接近 0 的情况 (射线平行)
    # 为了避免除以0，创建一个安全的 mask
    epsilon = 1e-8
    valid_mask = cross_norm > epsilon
    
    distances = np.zeros_like(cross_norm)
    
    # 正常情况：直接应用公式 d = |W0 . (D1 x D2)| / |D1 x D2|
    distances[valid_mask] = numerator[valid_mask] / cross_norm[valid_mask]
    
    # 平行情况 (极少见)：退化为点到直线的距离 |D1 x W0| / |D1| (由于D1归一化了，分母为1)
    if (~valid_mask).any():
        parallel_cross = np.cross(D1[~valid_mask], W0)
        distances[~valid_mask] = np.linalg.norm(parallel_cross, axis=1)
        
    return distances