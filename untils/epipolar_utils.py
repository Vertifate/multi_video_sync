import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# =========================================================================
#  环境路径修复区
# =========================================================================
# 定义项目根目录
PROJECT_ROOT_PATH = Path("/home/jinzhecui/Project/multi_video_sync-main")

# 1. 强制添加 LightGlue 路径
lightglue_dir = PROJECT_ROOT_PATH / "LightGlue"
if str(lightglue_dir) not in sys.path:
    sys.path.insert(0, str(lightglue_dir))

# 2. 强制添加 untils 路径 (优先加载项目中的 match_utils)
untils_dir = PROJECT_ROOT_PATH / "untils"
if str(untils_dir) not in sys.path:
    sys.path.insert(0, str(untils_dir))

print(f"[System] 环境路径已修正，正在从 {untils_dir} 加载模块...")
# =========================================================================

# 导入模块
try:
    from match_utils import MultiImageMatcher 
    from dataloader import qvec2rotmat, dataloader 
except ImportError as e:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from dataloader import qvec2rotmat, dataloader
    from match_utils import MultiImageMatcher
 
class EpipolarCalculator:
    def __init__(self, image1, camera1, image2, camera2):
        self.K = None
        self.R = None
        self.t = None
        self.F = None
        # 新增：用于存储计算出的对极线结果
        self.epilines = None 
        
        # 加载参数并计算基础矩阵
        self._load_params_from_objects(image1, camera1, image2, camera2)
        self._compute_fundamental_matrix() 
 
    def _load_params_from_objects(self, image1, camera1, image2, camera2):
        """
        修正版：正确计算从 Camera 1 到 Camera 2 的相对位姿
        """
        # 1. 获取 World -> Camera 的变换
        R1 = qvec2rotmat(image1.qvec)
        t1 = image1.tvec.reshape(3, 1)
        R2 = qvec2rotmat(image2.qvec)
        t2 = image2.tvec.reshape(3, 1)

        # 2. 计算相对旋转 R_rel = R2 * R1^T
        self.R = R2 @ R1.T
        
        # 3. 计算相对平移 t_rel = t2 - R_rel * t1
        self.t = t2 - self.R @ t1

        # 4. 构建内参矩阵 K (兼容多种模型)
        params = camera1.params
        model_name = camera1.model
        
        if model_name == "PINHOLE":
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        elif model_name == "SIMPLE_RADIAL":
            f, cx, cy = params[0], params[1], params[2]
            self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        elif model_name == "SIMPLE_PINHOLE":
            f, cx, cy = params[0], params[1], params[2]
            self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        else:
            print(f"[Warning] 未知相机模型 {model_name}，尝试通用解析...")
            if len(params) >= 4:
                self.K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
            else:
                self.K = np.array([[params[0], 0, params[1]], [0, params[0], params[2]], [0, 0, 1]])

    def _compute_fundamental_matrix(self):
        """计算基础矩阵 F"""
        tx, ty, tz = self.t.flatten()
        t_x = np.array([
            [0, -tz, ty],
            [tz, 0, -tx],
            [-ty, tx, 0]
        ])
        E = t_x @ self.R
        K_inv = np.linalg.inv(self.K)
        self.F = K_inv.T @ E @ K_inv

    def compute_epilines(self, points):
        """
        计算对极线，并保存到 self.epilines
        :param points: 输入点 (N, 2)
        :return: 对极线 (N, 3)
        """
        if self.F is None:
            raise RuntimeError("基础矩阵未计算")
        if len(points) == 0:
            self.epilines = np.array([])
            return self.epilines

        # OpenCV 需要 (N, 1, 2) 格式
        lines = cv2.computeCorrespondEpilines(points.reshape(-1, 1, 2), 1, self.F)
        
        # 保存结果到类属性
        self.epilines = lines.reshape(-1, 3)
        return self.epilines

    def visualize(self, points, epilines, img1_path, img2_path, save_path=None):
        """
        可视化：使用随机颜色绘制对应关系
        """
        POINT_RADIUS = 8
        LINE_THICKNESS = 2
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            print(f"[Error] 图片读取失败")
            return

        out_pts = img1.copy()
        out_lines = img2.copy()
        h, w = out_pts.shape[:2]

        print(f"[Info] 正在绘制 {len(points)} 条对极线 (颜色: 随机)...")

        np.random.seed(42)

        for r_line, pt in zip(epilines, points):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            pt_tuple = tuple(map(int, pt))
            
            # 左图：画点
            cv2.circle(out_pts, pt_tuple, POINT_RADIUS, color, -1)
            
            # 右图：画线
            a, b, c_val = r_line
            if abs(b) > 1e-10:
                x0, y0 = map(int, [0, -c_val/b])
                x1, y1 = map(int, [w, -(c_val + a*w)/b])
                cv2.line(out_lines, (x0, y0), (x1, y1), color, LINE_THICKNESS)
            else:
                x_val = int(-c_val/a)
                cv2.line(out_lines, (x_val, 0), (x_val, h), color, LINE_THICKNESS)

        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(out_pts, cv2.COLOR_BGR2RGB))
        plt.title(f'Feature Points (Random Color)', fontsize=14)
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(cv2.cvtColor(out_lines, cv2.COLOR_BGR2RGB))
        plt.title(f'Epipolar Lines (Random Color)', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[Info] 结果已保存: {save_path}")
        
        try:
            plt.show()
        except Exception:
            pass
        plt.close()

# ================= 主程序 =================
if __name__ == "__main__":
    # 配置区
    data_root = Path("/home/jinzhecui/Project/multi_sync/metashape") 
    target_frame = "frame000000"  # 目标帧
    cam_id_1 = "001"              # 左图 ID
    cam_id_2 = "019"              # 右图 ID

    print(f"[Info] 处理目标: {target_frame} | {cam_id_1} vs {cam_id_2}")

    # 1. 加载 COLMAP 数据
    loader1 = dataloader(str(data_root), cam_id_1) #
    loader2 = dataloader(str(data_root), cam_id_2) #

    if target_frame not in loader1.frame_images or target_frame not in loader2.frame_images:
        print(f"[Error] 找不到帧 {target_frame} 的数据")
        sys.exit(1)

    try:
        img_obj1 = list(loader1.frame_images[target_frame].values())[0] #
        img_obj2 = list(loader2.frame_images[target_frame].values())[0] #
    except IndexError:
        print("[Error] 数据提取失败")
        sys.exit(1)

    cam_obj1 = loader1.get_camera_params() #
    cam_obj2 = loader2.get_camera_params() #

    # 2. 确定图片路径
    img1_path = data_root / target_frame / "images" / img_obj1.name
    img2_path = data_root / target_frame / "images" / img_obj2.name

    if not img1_path.exists() or not img2_path.exists():
        print(f"[Error] 图片文件缺失")
        sys.exit(1)

    # 3. 特征匹配
    matcher = MultiImageMatcher() #
    print(f"[Info] 正在匹配特征...")
    _, matches_results = matcher.process_image_paths([str(img1_path), str(img2_path)])
    
    if not matches_results:
        print("[Error] 无匹配点")
        sys.exit(1)

    matches, m_kpts0, m_kpts1 = matches_results[0]
    points_for_epipolar = m_kpts0.cpu().numpy() 

    if len(points_for_epipolar) < 8:
        print("[Warning] 匹配点过少 (<8)")

    # 4. 计算对极几何
    calculator = EpipolarCalculator(img_obj1, cam_obj1, img_obj2, cam_obj2)
    
    # 这一步计算对极线，并自动保存到 calculator.epilines
    epilines = calculator.compute_epilines(points_for_epipolar) 
    
    print("[Success] 对极线计算完成。")
    print(f"[Info] 结果已保存至 calculator.epilines，共 {len(calculator.epilines)} 条线。")

    # 5. 可视化 (已注释，仅保留供以后测试)
    # output_filename = f"epipolar_random_{cam_id_1}_vs_{cam_id_2}.png"
    # calculator.visualize(
    #     points_for_epipolar, 
    #     epilines, 
    #     str(img1_path), 
    #     str(img2_path),
    #     save_path=output_filename
    # )