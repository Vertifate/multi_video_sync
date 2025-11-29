import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# 动态添加项目根目录到 sys.path，以便导入其他模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dataloader import qvec2rotmat
 
class EpipolarCalculator:
    def __init__(self, image1, camera1, image2, camera2):
        """
        初始化对极几何计算器。
        :param image1: 第一个视图的 Image 对象 (from dataloader)
        :param camera1: 第一个视图的 Camera 对象 (from dataloader)
        :param image2: 第二个视图的 Image 对象 (from dataloader)
        :param camera2: 第二个视图的 Camera 对象 (from dataloader)
        """
        
        # 内部数据
        self.K = None
        self.R = None
        self.t = None
        self.F = None
        
        # 初始化加载
        self._load_params_from_objects(image1, camera1, image2, camera2)
        self._compute_fundamental_matrix()
 
    def _load_params_from_objects(self, image1, camera1, image2, camera2):
        """从 dataloader 的对象中加载并计算相对位姿和内参"""
        # 提取相机1的世界位姿 (C1 -> W)
        R1 = qvec2rotmat(image1.qvec)
        t1 = image1.tvec.reshape(3, 1)

        # 提取相机2的世界位姿 (C2 -> W)
        R2 = qvec2rotmat(image2.qvec)
        t2 = image2.tvec.reshape(3, 1)

        # 计算从相机1到相机2的相对位姿 (C1 -> C2)
        # R_rel = R2.T @ R1
        # t_rel = R2.T @ (t1 - t2)
        # 我们需要从相机2到相机1的相对位姿 (C2 -> C1) 来计算点在1中，线在2中的情况
        self.R = R1.T @ R2
        self.t = R1.T @ (t2 - t1)

        # 提取相机1的内参矩阵 K
        # 这里假设为PINHOLE模型 (fx, fy, cx, cy)
        if camera1.model != "PINHOLE":
            print(f"[Warning] Camera model is {camera1.model}, not PINHOLE. Assuming fx,fy,cx,cy params.")
        fx, fy, cx, cy = camera1.params
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        print("[Info] 相机参数加载并计算成功。")

    def _compute_fundamental_matrix(self):
        """计算基础矩阵 F"""
        # [t]x 反对称矩阵
        tx, ty, tz = self.t.flatten()
        t_x = np.array([
            [0, -tz, ty],
            [tz, 0, -tx],
            [-ty, tx, 0]
        ])
        
        # E = [t]x * R
        E = t_x @ self.R
        
        # F = K^(-T) * E * K^(-1)
        K_inv = np.linalg.inv(self.K)
        self.F = K_inv.T @ E @ K_inv

    def compute_epilines(self, points):
        """
        为给定的点计算对极线。
        :param points: 特征点数组，shape 为 (N, 2)
        :return: 对极线数组，shape 为 (N, 3)，每条线为 (a, b, c) 对应 ax+by+c=0
        """
        if self.F is None:
            raise RuntimeError("基础矩阵未计算，无法继续。")
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("输入点必须是 shape 为 (N, 2) 的 Numpy 数组。")
        if len(points) == 0:
            print("[Warn] 点列表为空，返回空数组。")
            return np.array([])

        # cv2.computeCorrespondEpilines 要求输入 shape 为 (N, 1, 2)
        lines = cv2.computeCorrespondEpilines(points.reshape(-1, 1, 2), 1, self.F)
        return lines.reshape(-1, 3)

    def visualize(self, points, epilines, img1_path, img2_path):
        """
        可视化特征点和对应的对极线。
        :param points: 特征点数组, shape (N, 2)
        :param epilines: 对极线数组, shape (N, 3)
        :param img1_path: 左图路径 (用于绘制点)
        :param img2_path: 右图路径 (用于绘制线)
        """
        # 视觉样式 (粗线大点)
        POINT_RADIUS = 12
        LINE_THICKNESS = 4

        # 读取图像
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print("[Error] 无法读取图片，请检查路径。使用黑色背景代替。")
            img1 = np.zeros((480, 640, 3), dtype=np.uint8)
            img2 = np.zeros((480, 640, 3), dtype=np.uint8)

        out_pts = img1.copy()
        out_lines = img2.copy()
        h, w = out_pts.shape[:2]

        np.random.seed(42) # 固定颜色

        print(f"[Info] 正在绘制全部 {len(points)} 条对极线...")

        for r_line, pt in zip(epilines, points):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            pt_tuple = tuple(map(int, pt))
            
            # --- 左图：画点 ---
            cv2.circle(out_pts, pt_tuple, POINT_RADIUS, color, -1)
            cv2.circle(out_pts, pt_tuple, POINT_RADIUS, (255, 255, 255), 2)
            
            # --- 右图：画线 ---
            a, b, c_val = r_line
            if abs(b) > 1e-10:
                x0, y0 = map(int, [0, -c_val/b])
                x1, y1 = map(int, [w, -(c_val + a*w)/b])
                cv2.line(out_lines, (x0, y0), (x1, y1), color, LINE_THICKNESS)
            else:
                x_val = int(-c_val/a)
                cv2.line(out_lines, (x_val, 0), (x_val, h), color, LINE_THICKNESS)

        # 显示
        plt.figure(figsize=(16, 8))
        
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(out_pts, cv2.COLOR_BGR2RGB))
        plt.title(f'Input Points ({len(points)})', fontsize=14)
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(cv2.cvtColor(out_lines, cv2.COLOR_BGR2RGB))
        plt.title(f'Epipolar Lines ({len(points)})', fontsize=14)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# ================= 使用示例 =================
if __name__ == "__main__":
    # 这是一个完整的集成示例，演示如何结合 dataloader 和 match_utils
    from dataloader import DataLoader
    from utils.match_utils import MultiImageMatcher
    from pathlib import Path

    # --- 1. 使用 DataLoader 加载数据 ---
    # 指定包含 frameXXXXXX 目录的数据根目录
    data_root = Path(project_root) / "metashape"
    frame_name = "frame000026"
    frame_dir = data_root / frame_name

    loader = DataLoader(str(data_root))
    loader.load_colmap_data(frame_dir, frame_name) # 只加载特定一帧的数据

    # 获取该帧的所有图像信息
    images_in_frame = loader.frame_images[frame_name]
    cameras = loader.cameras

    if len(images_in_frame) < 2:
        print("错误：需要至少两张图像来进行对极几何计算。")
    else:
        # 选择前两张图进行演示
        img_info1 = list(images_in_frame.values())[0]
        img_info2 = list(images_in_frame.values())[1]
        cam1 = cameras[img_info1.camera_id]
        cam2 = cameras[img_info2.camera_id]
        
        img1_path = frame_dir / "images" / img_info1.name
        img2_path = frame_dir / "images" / img_info2.name

        # --- 2. 使用 MultiImageMatcher 匹配特征点 ---
        matcher = MultiImageMatcher()
        _, matches_results = matcher.process_image_paths([str(img1_path), str(img2_path)])
        
        # 提取第一对匹配的结果 (img1 vs img2)
        # 我们需要 img1 上的点 (m_kpts0) 来计算在 img2 上的对极线
        matches, m_kpts0, m_kpts1 = matches_results[0]
        points_to_compute = m_kpts0.numpy() # 转换为 numpy
        print(f"[Info] 从匹配结果中提取了 {len(points_to_compute)} 个特征点。")

        # --- 3. 初始化 EpipolarCalculator 并计算 ---
        calculator = EpipolarCalculator(img_info1, cam1, img_info2, cam2)
        
        # 计算对极线
        epilines = calculator.compute_epilines(points_to_compute)
        print("[Info] 对极线计算完成。")

        # --- 4. 可选的可视化 ---
        visualize_results = True  # 修改这里来控制是否显示
        if visualize_results:
            print("\n[Info] 正在启动可视化...")
            # 注意：我们将 img1 上的点 (m_kpts0) 和 img2 上的对极线一起可视化
            calculator.visualize(points_to_compute, epilines, str(img1_path), str(img2_path))
