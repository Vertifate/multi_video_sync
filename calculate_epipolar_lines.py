import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

class EpipolarVisualizer:
    def __init__(self, camera_json, points_json, img1_path, img2_path):
        """
        初始化可视化器
        :param camera_json: 相机参数文件路径 (K, R, t)
        :param points_json: 特征点文件路径 (List of points)
        :param img1_path: 左图路径
        :param img2_path: 右图路径
        """
        self.camera_json = camera_json
        self.points_json = points_json
        self.img1_path = img1_path
        self.img2_path = img2_path
        
        # 内部数据
        self.K = None
        self.R = None
        self.t = None
        self.points = None
        self.F = None
        self.epilines = None
        
        # 视觉样式 (粗线大点)
        self.POINT_RADIUS = 12
        self.LINE_THICKNESS = 4
        
        # 初始化加载
        self._load_camera_params()
        self._load_points()
        self._compute_fundamental_matrix()

    def _load_camera_params(self):
        """加载相机参数 K, R, t"""
        if not os.path.exists(self.camera_json):
            raise FileNotFoundError(f"找不到相机文件: {self.camera_json}")
            
        with open(self.camera_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        try:
            self.K = np.array(data['K'], dtype=np.float64)
            self.R = np.array(data['R'], dtype=np.float64)
            # 兼容一维数组或二维数组的 t
            self.t = np.array(data['t'], dtype=np.float64).reshape(3, 1)
            print("[Info] 相机参数加载成功。")
        except KeyError as e:
            raise ValueError(f"相机 JSON 缺少键值: {e}")

    def _load_points(self):
        """加载特征点坐标"""
        if not os.path.exists(self.points_json):
            raise FileNotFoundError(f"找不到点文件: {self.points_json}")
            
        with open(self.points_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 转换为 numpy 数组
        self.points = np.array(data, dtype=np.float64)
        print(f"[Info] 加载了 {len(self.points)} 个特征点。")

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

    def run(self):
        """运行计算并可视化"""
        if self.points is None or len(self.points) == 0:
            print("[Warn]点列表为空，无法绘制。")
            return

        # 1. 计算所有点的对极线
        # cv2.computeCorrespondEpilines 要求输入 shape 为 (N, 1, 2)
        lines = cv2.computeCorrespondEpilines(self.points.reshape(-1, 1, 2), 1, self.F)
        self.epilines = lines.reshape(-1, 3)
        
        # 2. 读取图像
        img1 = cv2.imread(self.img1_path)
        img2 = cv2.imread(self.img2_path)
        
        if img1 is None or img2 is None:
            print("[Error] 无法读取图片，请检查路径。使用黑色背景代替。")
            img1 = np.zeros((480, 640, 3), dtype=np.uint8)
            img2 = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 3. 绘制
        self._draw_and_show(img1, img2)

    def _draw_and_show(self, img1, img2):
        out_pts = img1.copy()
        out_lines = img2.copy()
        h, w = out_pts.shape[:2]
        
        np.random.seed(42) # 固定颜色
        
        print(f"[Info] 正在绘制全部 {len(self.points)} 条对极线...")

        for r_line, pt in zip(self.epilines, self.points):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            pt_tuple = tuple(map(int, pt))
            
            # --- 左图：画点 ---
            cv2.circle(out_pts, pt_tuple, self.POINT_RADIUS, color, -1)
            cv2.circle(out_pts, pt_tuple, self.POINT_RADIUS, (255, 255, 255), 2)
            
            # --- 右图：画线 ---
            a, b, c_val = r_line
            if abs(b) > 1e-10:
                x0, y0 = map(int, [0, -c_val/b])
                x1, y1 = map(int, [w, -(c_val + a*w)/b])
                cv2.line(out_lines, (x0, y0), (x1, y1), color, self.LINE_THICKNESS)
            else:
                x_val = int(-c_val/a)
                cv2.line(out_lines, (x_val, 0), (x_val, h), color, self.LINE_THICKNESS)

        # 显示
        plt.figure(figsize=(16, 8))
        
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(out_pts, cv2.COLOR_BGR2RGB))
        plt.title(f'Input Points ({len(self.points)})', fontsize=14)
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(cv2.cvtColor(out_lines, cv2.COLOR_BGR2RGB))
        plt.title(f'Epipolar Lines ({len(self.points)})', fontsize=14)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# ================= 使用示例 =================
if __name__ == "__main__":
    # 定义所有文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 输入文件
    camera_file = os.path.join(current_dir, 'camera.json')
    points_file = os.path.join(current_dir, 'points.json')
    img_left = os.path.join(current_dir, 'left.jpg')
    img_right = os.path.join(current_dir, 'right.jpg')
    
    # 检查文件是否存在
    if os.path.exists(camera_file) and os.path.exists(points_file):
        viz = EpipolarVisualizer(camera_file, points_file, img_left, img_right)
        viz.run()
    else:
        print("错误：请确保 camera.json 和 points.json 都在当前目录下。")