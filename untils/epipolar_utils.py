# import os
# from pathlib import Path
# import sys

# force_qt_ui = os.environ.get("FORCE_QT_UI", "0") == "1"
# force_headless = os.environ.get("FORCE_HEADLESS", "0") == "1"
# has_display = os.environ.get("DISPLAY", "") != ""
 
# import cv2
# import torch 
# import numpy as np

# # ================= 路径修复区域 =================
# # 原来的相对路径逻辑在下载目录运行会失效，改为使用绝对路径
# # 指向: /home/jinzhecui/Project/multi_video_sync-main/LightGlue
# lightglue_path = Path("/home/jinzhecui/Project/multi_video_sync-main/LightGlue")

# if lightglue_path.is_dir():
#     # 将 LightGlue 所在目录加入 python 搜索路径
#     if str(lightglue_path) not in sys.path:
#         sys.path.insert(0, str(lightglue_path))
#     print(f"[Info] LightGlue path set to: {lightglue_path}")
# else:
#     # 尝试指向父目录 (有时候 LightGlue 包直接就在 multi_video_sync-main 下面，没有嵌套一层 LightGlue 文件夹)
#     fallback_path = lightglue_path.parent
#     if (fallback_path / "lightglue").is_dir():
#          if str(fallback_path) not in sys.path:
#             sys.path.insert(0, str(fallback_path))
#          print(f"[Info] LightGlue path set to: {fallback_path}")
#     else:
#         raise FileNotFoundError(f"LightGlue directory not found at {lightglue_path}")
# # ===============================================

# # 现在路径设置好了，可以安全导入了
# try:
#     from lightglue import LightGlue, SuperPoint, DISK
#     from lightglue.utils import load_image, rbd
# except ImportError as e:
#     print(f"[Error] 导入 LightGlue 失败: {e}")
#     print("请确认 LightGlue 文件夹内包含 lightglue 包（即文件夹内有 __init__.py）")
#     sys.exit(1)
 

# class MultiImageMatcher:
#     def __init__(self, device=None):
#         """
#         初始化多图像匹配器
        
#         Args:
#             device: 计算设备 ('cuda', 'cpu' 等)
#         """
#         if device is None:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             self.device = device
            
#         # 加载特征提取器和匹配器
#         try:
#             self.extractor = SuperPoint(max_num_keypoints=2048000).eval().to(self.device)
#             self.matcher = LightGlue(features="superpoint").eval().to(self.device)
#         except Exception as e:
#             print(f"[Error] 模型加载失败: {e}")
#             raise

#     def load_images(self, image_paths):
#         """
#         加载图像列表
        
#         Args:
#             image_paths: 图像路径列表
            
#         Returns:
#             list: 加载的图像列表
#         """
#         images = []
#         for path in image_paths:
#             # 确保路径是字符串
#             path_str = str(path)
#             if not os.path.exists(path_str):
#                 raise FileNotFoundError(f"Image not found: {path_str}")
#             image = load_image(path_str)
#             images.append(image)
#         return images
    
#     def match_images(self, images):
#         """
#         将后续图像与第一张图像进行匹配
        
#         Args:
#             images: 图像列表，第一张图像为主视角
            
#         Returns:
#             list: 每个元素是一个元组 (matches, m_kpts0, m_kpts1)，表示与主图像的匹配结果
#         """
#         if len(images) < 2:
#             raise ValueError("需要至少两张图像来进行匹配")
            
#         # 第一张图像作为主视角
#         main_image = images[0]
#         main_feats = self.extractor.extract(main_image.to(self.device))
        
#         results = []
        
#         # 对于后续的每张图像，与主图像进行匹配
#         for i in range(1, len(images)):
#             # 提取当前图像的特征
#             curr_feats = self.extractor.extract(images[i].to(self.device))
            
#             # 匹配主图像和当前图像的特征
#             matches_result = self.matcher({"image0": main_feats, "image1": curr_feats})
#             main_feats_copy, curr_feats_copy, matches_result = [rbd(x) for x in [main_feats, curr_feats, matches_result]]
            
#             # 获取匹配的关键点
#             main_kpts = main_feats_copy["keypoints"]
#             curr_kpts = curr_feats_copy["keypoints"]
#             matches = matches_result["matches"]
#             m_kpts0 = main_kpts[matches[..., 0]]
#             m_kpts1 = curr_kpts[matches[..., 1]]
            
#             # 保存结果
#             results.append((matches, m_kpts0, m_kpts1))
            
#         return results
    
#     def process_image_paths(self, image_paths):
#         """
#         处理图像路径列表，加载图像并执行匹配
        
#         Args:
#             image_paths: 图像路径列表
            
#         Returns:
#             tuple: (images, matches_results) 图像列表和匹配结果
#         """
#         # 加载所有图像
#         images = self.load_images(image_paths)
        
#         # 执行匹配
#         matches_results = self.match_images(images)
        
#         return images, matches_results
    
#     def visualize_matches(self, images, matches_results, image_paths=None, save_dir=None):
#         """
#         使用OpenCV可视化匹配结果。
        
#         Args:
#             images: 图像列表
#             matches_results: 匹配结果列表
#             image_paths: 图像路径列表（可选）
#             save_dir: 保存可视化结果的目录路径（可选）
#         """
#         num_matches = len(matches_results)
        
#         if save_dir:
#             save_path = Path(save_dir)
#             save_path.mkdir(exist_ok=True, parents=True)
        
#         for i in range(num_matches):
#             matches, m_kpts0, m_kpts1 = matches_results[i]
#             print(f'第{i+1}张图像与主图像之间找到 {len(matches)} 个匹配点')

#             # 准备OpenCV需要的数据格式
#             # 将 PyTorch 的 RGB tensor 转换为 OpenCV 的 BGR numpy 数组
#             main_image_bgr = cv2.cvtColor((images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
#             curr_image_bgr = cv2.cvtColor((images[i+1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
#             # 将torch tensor关键点转换为cv2.KeyPoint对象列表
#             kpts0_cv = [cv2.KeyPoint(p[0].item(), p[1].item(), 1) for p in m_kpts0]
#             kpts1_cv = [cv2.KeyPoint(p[0].item(), p[1].item(), 1) for p in m_kpts1]
            
#             # 创建DMatch对象
#             dmatches = [cv2.DMatch(_queryIdx=j, _trainIdx=j, _distance=0) for j in range(len(kpts0_cv))]
            
#             # 使用cv2.drawMatches绘制匹配
#             match_img = cv2.drawMatches(main_image_bgr, kpts0_cv, curr_image_bgr, kpts1_cv, dmatches, None, 
#                                         matchColor=(0, 255, 0),  # BGR for lime green
#                                         singlePointColor=None, 
#                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#             window_title = "Matches"
#             if image_paths:
#                 # 确保取文件名而不是完整路径
#                 name1 = Path(image_paths[0]).name
#                 name2 = Path(image_paths[i+1]).name
#                 window_title = f"{name1} vs {name2}"

#             if save_dir:
#                 name1_stem = Path(image_paths[0]).stem if image_paths else "img0"
#                 name2_stem = Path(image_paths[i+1]).stem if image_paths else f"img{i+1}"
#                 output_filename = save_path / f"match_{name1_stem}_vs_{name2_stem}.png"
#                 cv2.imwrite(str(output_filename), match_img)
#                 print(f"匹配结果已保存到: {output_filename}")

#             # --- 调整图像大小以便于显示 ---
#             max_display_width = 1920  # 设置显示窗口的最大宽度
#             h, w = match_img.shape[:2]
            
#             if w > max_display_width:
#                 scale = max_display_width / w
#                 new_w = int(w * scale)
#                 new_h = int(h * scale)
#                 display_img = cv2.resize(match_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
#             else:
#                 display_img = match_img

#             cv2.imshow(window_title, display_img)
#             key = cv2.waitKey(0) # 等待按键
#             if key == ord('q') or key == 27: # 如果是'q'或'Esc'
#                 break # 退出循环
#         cv2.destroyAllWindows()


# # 原始示例代码的使用方式
# if __name__ == "__main__":
#     # 设置设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 创建匹配器实例
#     matcher = MultiImageMatcher(device)
    
#     # 测试路径 - 请根据实际情况修改
#     data_dir = Path("/home/jinzhecui/Project/multi_sync/metashape/frame000026/images")
#     if data_dir.exists():
#         image_paths = sorted(list(data_dir.glob("*.jpg")))  # 加载所有图像
        
#         if len(image_paths) < 2:
#             print("需要至少两张图像来进行匹配")
#         else:
#             # 仅取前两张测试
#             test_paths = image_paths[:2]
#             print(f"Testing with: {test_paths}")
            
#             # 处理图像
#             images, matches_results = matcher.process_image_paths(test_paths)
            
#             # 可视化结果（可选）
#             matcher.visualize_matches(images, matches_results, test_paths)
#     else:
#         print(f"测试目录不存在: {data_dir}")


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import sys
# from pathlib import Path

# # =========================================================================
# #  环境路径修复区
# # =========================================================================
# # 定义项目根目录
# PROJECT_ROOT_PATH = Path("/home/jinzhecui/Project/multi_video_sync-main")

# # 1. 强制添加 LightGlue 路径
# lightglue_dir = PROJECT_ROOT_PATH / "LightGlue"
# if str(lightglue_dir) not in sys.path:
#     sys.path.insert(0, str(lightglue_dir))

# # 2. 强制添加 untils 路径 (优先加载项目中的 match_utils)
# untils_dir = PROJECT_ROOT_PATH / "untils"
# if str(untils_dir) not in sys.path:
#     sys.path.insert(0, str(untils_dir))

# print(f"[System] 环境路径已修正，正在从 {untils_dir} 加载模块...")
# # =========================================================================

# # 导入模块
# try:
#     from match_utils import MultiImageMatcher 
#     from dataloader import qvec2rotmat, dataloader 
# except ImportError as e:
#     # 备用方案：尝试从当前目录加载
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     if current_dir not in sys.path:
#         sys.path.append(current_dir)
#     from dataloader import qvec2rotmat, dataloader
#     from match_utils import MultiImageMatcher
 
# class EpipolarCalculator:
#     def __init__(self, image1, camera1, image2, camera2):
#         self.K = None
#         self.R = None
#         self.t = None
#         self.F = None
#         # 加载参数并计算基础矩阵
#         self._load_params_from_objects(image1, camera1, image2, camera2)
#         self._compute_fundamental_matrix() 
 
#     def _load_params_from_objects(self, image1, camera1, image2, camera2):
#         """
#         修正版：正确计算从 Camera 1 到 Camera 2 的相对位姿
#         """
#         # 1. 获取 World -> Camera 的变换
#         R1 = qvec2rotmat(image1.qvec) 
#         t1 = image1.tvec.reshape(3, 1) 
#         R2 = qvec2rotmat(image2.qvec) 
#         t2 = image2.tvec.reshape(3, 1) 

#         # 2. 计算相对旋转 R_rel = R2 * R1^T
#         # (推导: P2 = R2*Pw + t2, P1 = R1*Pw + t1 => Pw = R1^T*(P1-t1)
#         #  => P2 = R2*R1^T*P1 + (t2 - R2*R1^T*t1))
#         self.R = R2 @ R1.T
        
#         # 3. 计算相对平移 t_rel = t2 - R_rel * t1
#         self.t = t2 - self.R @ t1

#         # 4. 构建内参矩阵 K (兼容多种模型)
#         params = camera1.params
#         model_name = camera1.model
        
#         if model_name == "PINHOLE":
#             # fx, fy, cx, cy
#             fx, fy, cx, cy = params[0], params[1], params[2], params[3]
#             self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
#         elif model_name == "SIMPLE_RADIAL":
#             # f, cx, cy, k
#             f, cx, cy = params[0], params[1], params[2]
#             self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            
#         elif model_name == "SIMPLE_PINHOLE":
#             # f, cx, cy
#             f, cx, cy = params[0], params[1], params[2]
#             self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            
#         else:
#             # 默认尝试取前4个参数，假设是 fx, fy, cx, cy 或者是 f, cx, cy
#             print(f"[Warning] 未知相机模型 {model_name}，尝试通用解析...")
#             if len(params) >= 4:
#                 self.K = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1]])
#             else:
#                 self.K = np.array([[params[0], 0, params[1]], [0, params[0], params[2]], [0, 0, 1]])

#     def _compute_fundamental_matrix(self):
#         """计算基础矩阵 F = K^-T * [t]x * R * K^-1"""
#         tx, ty, tz = self.t.flatten()
#         # 构造反对称矩阵 [t]x
#         t_x = np.array([
#             [0, -tz, ty],
#             [tz, 0, -tx],
#             [-ty, tx, 0]
#         ])
        
#         # 本质矩阵 E
#         E = t_x @ self.R
        
#         # 基础矩阵 F
#         K_inv = np.linalg.inv(self.K)
#         self.F = K_inv.T @ E @ K_inv 

#     def compute_epilines(self, points):
#         """计算对极线"""
#         if self.F is None:
#             raise RuntimeError("基础矩阵未计算")
#         if len(points) == 0:
#             return np.array([])
#         # OpenCV 需要 (N, 1, 2) 格式
#         lines = cv2.computeCorrespondEpilines(points.reshape(-1, 1, 2), 1, self.F)
#         return lines.reshape(-1, 3) 

#     def visualize(self, points, epilines, img1_path, img2_path, save_path=None):
#         """只画绿点和绿线"""
#         POINT_RADIUS = 8
#         LINE_THICKNESS = 2
#         COLOR_GREEN = (0, 255, 0) # BGR

#         img1 = cv2.imread(img1_path)
#         img2 = cv2.imread(img2_path)

#         if img1 is None or img2 is None:
#             print(f"[Error] 图片读取失败")
#             return

#         out_pts = img1.copy()
#         out_lines = img2.copy()
#         h, w = out_pts.shape[:2]

#         print(f"[Info] 正在绘制 {len(points)} 条对极线...")

#         for r_line, pt in zip(epilines, points):
#             color = COLOR_GREEN
#             pt_tuple = tuple(map(int, pt))
            
#             # 左图：画绿点
#             cv2.circle(out_pts, pt_tuple, POINT_RADIUS, color, -1)
            
#             # 右图：画绿线 (ax + by + c = 0)
#             a, b, c_val = r_line
#             if abs(b) > 1e-10:
#                 x0, y0 = map(int, [0, -c_val/b])
#                 x1, y1 = map(int, [w, -(c_val + a*w)/b])
#                 cv2.line(out_lines, (x0, y0), (x1, y1), color, LINE_THICKNESS)
#             else:
#                 x_val = int(-c_val/a)
#                 cv2.line(out_lines, (x_val, 0), (x_val, h), color, LINE_THICKNESS)

#         # Matplotlib 可视化
#         plt.figure(figsize=(16, 8))
        
#         plt.subplot(121)
#         plt.imshow(cv2.cvtColor(out_pts, cv2.COLOR_BGR2RGB))
#         plt.title(f'Feature Points (Green)', fontsize=14)
#         plt.axis('off')

#         plt.subplot(122)
#         plt.imshow(cv2.cvtColor(out_lines, cv2.COLOR_BGR2RGB))
#         plt.title(f'Epipolar Lines (Green)', fontsize=14)
#         plt.axis('off')

#         plt.tight_layout()
        
#         if save_path:
#             plt.savefig(save_path, dpi=150, bbox_inches='tight')
#             print(f"[Info] 结果已保存: {save_path}")
        
#         try:
#             plt.show()
#         except Exception:
#             pass
#         plt.close()

# # ================= 主程序 =================
# if __name__ == "__main__":
#     # 配置区
#     data_root = Path("/home/jinzhecui/Project/multi_sync/metashape") 
#     target_frame = "frame000000"  # 目标帧
#     cam_id_1 = "001"              # 左图 ID
#     cam_id_2 = "019"              # 右图 ID

#     print(f"[Info] 处理目标: {target_frame} | {cam_id_1} vs {cam_id_2}")

#     # 1. 加载 COLMAP 数据
#     loader1 = dataloader(str(data_root), cam_id_1) #
#     loader2 = dataloader(str(data_root), cam_id_2) #

#     if target_frame not in loader1.frame_images or target_frame not in loader2.frame_images:
#         print(f"[Error] 找不到帧 {target_frame} 的数据")
#         sys.exit(1)

#     try:
#         img_obj1 = list(loader1.frame_images[target_frame].values())[0] #
#         img_obj2 = list(loader2.frame_images[target_frame].values())[0] #
#     except IndexError:
#         print("[Error] 数据提取失败")
#         sys.exit(1)

#     cam_obj1 = loader1.get_camera_params() #
#     cam_obj2 = loader2.get_camera_params() #

#     # 2. 确定图片路径
#     img1_path = data_root / target_frame / "images" / img_obj1.name
#     img2_path = data_root / target_frame / "images" / img_obj2.name

#     if not img1_path.exists() or not img2_path.exists():
#         print(f"[Error] 图片文件缺失")
#         sys.exit(1)

#     # 3. 特征匹配
#     matcher = MultiImageMatcher() #
#     print(f"[Info] 正在匹配特征...")
#     _, matches_results = matcher.process_image_paths([str(img1_path), str(img2_path)])
    
#     if not matches_results:
#         print("[Error] 无匹配点")
#         sys.exit(1)

#     matches, m_kpts0, m_kpts1 = matches_results[0]
#     points_for_epipolar = m_kpts0.cpu().numpy() 

#     if len(points_for_epipolar) < 8:
#         print("[Warning] 匹配点过少 (<8)")

#     # 4. 计算对极几何 (使用修正后的数学模型)
#     calculator = EpipolarCalculator(img_obj1, cam_obj1, img_obj2, cam_obj2)
#     epilines = calculator.compute_epilines(points_for_epipolar) #

#     # 5. 可视化
#     output_filename = f"epipolar_green_{cam_id_1}_vs_{cam_id_2}.png"
#     calculator.visualize(
#         points_for_epipolar, 
#         epilines, 
#         str(img1_path), 
#         str(img2_path),
#         save_path=output_filename
#     )


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
            # fx, fy, cx, cy
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
        elif model_name == "SIMPLE_RADIAL":
            # f, cx, cy
            f, cx, cy = params[0], params[1], params[2]
            self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            
        elif model_name == "SIMPLE_PINHOLE":
            # f, cx, cy
            f, cx, cy = params[0], params[1], params[2]
            self.K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            
        else:
            # 默认尝试通用解析
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
        """计算对极线"""
        if self.F is None:
            raise RuntimeError("基础矩阵未计算")
        if len(points) == 0:
            return np.array([])
        lines = cv2.computeCorrespondEpilines(points.reshape(-1, 1, 2), 1, self.F)
        return lines.reshape(-1, 3)

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

        # 设置随机种子，保证每次运行结果颜色一致，方便对比
        np.random.seed(42)

        for r_line, pt in zip(epilines, points):
            # --- 修改处：生成随机颜色 (B, G, R) ---
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            pt_tuple = tuple(map(int, pt))
            
            # 左图：画带颜色的点
            cv2.circle(out_pts, pt_tuple, POINT_RADIUS, color, -1)
            
            # 右图：画相同颜色的线 (ax + by + c = 0)
            a, b, c_val = r_line
            if abs(b) > 1e-10:
                x0, y0 = map(int, [0, -c_val/b])
                x1, y1 = map(int, [w, -(c_val + a*w)/b])
                cv2.line(out_lines, (x0, y0), (x1, y1), color, LINE_THICKNESS)
            else:
                x_val = int(-c_val/a)
                cv2.line(out_lines, (x_val, 0), (x_val, h), color, LINE_THICKNESS)

        # Matplotlib 可视化
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

    # 4. 计算对极几何 (使用修正后的数学模型)
    calculator = EpipolarCalculator(img_obj1, cam_obj1, img_obj2, cam_obj2)
    epilines = calculator.compute_epilines(points_for_epipolar) #

    # 5. 可视化
    output_filename = f"epipolar_random_{cam_id_1}_vs_{cam_id_2}.png"
    calculator.visualize(
        points_for_epipolar, 
        epilines, 
        str(img1_path), 
        str(img2_path),
        save_path=output_filename
    )