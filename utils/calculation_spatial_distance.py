import os
import struct
import numpy as np

class RayUtils:
    def __init__(self, project_root):
        """
        初始化工具类
        :param project_root: 项目的根目录路径 (e.g. "/home/user/Project/...")
        """
        self.project_root = project_root

    # ==========================================
    # 核心数学计算 (无需修改)
    # ==========================================
    @staticmethod
    def qvec2rotmat(qvec):
        w, x, y, z = qvec
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
        ])

    @staticmethod
    def pixel_to_ray(u, v, K, R, T):
        R_inv = R.T
        origin = -R_inv @ T
        pixel_h = np.array([u, v, 1.0])
        K_inv = np.linalg.inv(K)
        direction_cam = K_inv @ pixel_h
        direction_world = R_inv @ direction_cam
        direction_world = direction_world / np.linalg.norm(direction_world)
        return origin, direction_world

    @staticmethod
    def compute_skew_line_distance(O1, D1, O2, D2):
        W0 = O1 - O2
        cross_prod = np.cross(D1, D2)
        cross_norm = np.linalg.norm(cross_prod)
        if cross_norm < 1e-6:
            return np.linalg.norm(np.cross(D1, W0))
        return np.abs(np.dot(W0, cross_prod)) / cross_norm

    # ==========================================
    # COLMAP 参数读取
    # ==========================================
    def read_colmap_params(self, model_path, target_img_name):
        cameras_file = os.path.join(model_path, "cameras.bin")
        images_file = os.path.join(model_path, "images.bin")
        
        if not os.path.exists(cameras_file) or not os.path.exists(images_file):
            # 静默失败或打印简单的错误，避免刷屏
            return None

        # 1. 读取内参 K
        cameras = {}
        with open(cameras_file, "rb") as fid:
            num_cameras = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_cameras):
                cam_props = struct.unpack("<iiQQ", fid.read(24))
                cam_id, model_id = cam_props[0], cam_props[1]
                param_lens = {0:3, 1:4, 2:4, 3:5, 4:8, 5:9} 
                p_len = param_lens.get(model_id, 4) 
                params = struct.unpack("<" + "d" * p_len, fid.read(8 * p_len))
                
                K = np.eye(3)
                if model_id in [1, 4]: 
                    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = params[0], params[1], params[2], params[3]
                elif model_id in [0, 2]:
                    K[0, 0], K[1, 1], K[0, 2], K[1, 2] = params[0], params[0], params[1], params[2]
                cameras[cam_id] = K

        # 2. 读取外参 R, T
        with open(images_file, "rb") as fid:
            num_images = struct.unpack("<Q", fid.read(8))[0]
            for _ in range(num_images):
                binary_line = fid.read(64)
                image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack("<idddddddi", binary_line)
                
                name_chars = []
                while True:
                    c = fid.read(1)
                    if c == b"\0": break
                    name_chars.append(c.decode("utf-8"))
                name = "".join(name_chars)
                
                num_points2d = struct.unpack("<Q", fid.read(8))[0]
                fid.seek(24 * num_points2d, 1) 
                
                if name == target_img_name:
                    R = self.qvec2rotmat([qw, qx, qy, qz])
                    T = np.array([tx, ty, tz])
                    return {'K': cameras[camera_id], 'R': R, 'T': T}
        return None

    # ==========================================
    # 对外接口
    # ==========================================
    def calculate_distance(self, frame_num, cam1_id, pt1, cam2_id, pt2):
        """
        :param frame_num: 帧号 (int)
        :param cam1_id: 相机1 ID (int)
        :param pt1: (u, v)
        :param cam2_id: 相机2 ID (int)
        :param pt2: (u, v)
        :return: float (distance) or None
        """
        # 构建路径： root/frameXXXXXX/sparse/0
        frame_str = f"frame{frame_num:06d}"
        model_path = os.path.join(self.project_root, frame_str, "sparse", "0")
        
        # 构建文件名：XXX.jpg
        img1_name = f"{cam1_id:03d}.jpg"
        img2_name = f"{cam2_id:03d}.jpg"
        
        # 读取
        params1 = self.read_colmap_params(model_path, img1_name)
        params2 = self.read_colmap_params(model_path, img2_name)
        
        if params1 is None or params2 is None:
            return None # 读取失败

        # 计算
        O1, D1 = self.pixel_to_ray(pt1[0], pt1[1], **params1)
        O2, D2 = self.pixel_to_ray(pt2[0], pt2[1], **params2)

        return self.compute_skew_line_distance(O1, D1, O2, D2)
    
def main():
    # --- 1. 这里定义你的路径 ---
    my_project_path = "/home/jinzhecui/Project/multi_sync/metashape"
    
    # --- 2. 初始化工具 ---
    ray_tool = RayUtils(project_root=my_project_path)
    
    # --- 3. 准备输入数据 ---
    frame_idx = 0          # frame000000
    
    # 第一组数据
    cam1 = 1               # 001.jpg
    point1 = (1843.5, 940.2)
    
    # 第二组数据
    cam2 = 31              # 031.jpg
    point2 = (1920.0, 850.5)
    
    # --- 4. 计算并获取结果 ---
    dist = ray_tool.calculate_distance(frame_idx, cam1, point1, cam2, point2)
    
    if dist is not None:
        print(dist)  # 直接输出数值
    else:
        print("Error: Could not calculate distance (check paths/IDs).")

if __name__ == "__main__":
    main()