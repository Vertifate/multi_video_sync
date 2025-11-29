import os
from pathlib import Path
import sys

force_qt_ui = os.environ.get("FORCE_QT_UI", "0") == "1"
force_headless = os.environ.get("FORCE_HEADLESS", "0") == "1"
has_display = os.environ.get("DISPLAY", "") != ""
 

import cv2
import torch 
import numpy as np

# 添加LightGlue到路径
project_root = Path(__file__).resolve().parent.parent
lightglue_path = project_root / "LightGlue"
if lightglue_path.is_dir():
    sys.path.insert(0, str(lightglue_path))
else:
    raise FileNotFoundError(f"LightGlue directory not found at {lightglue_path}")

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
from lightglue.utils import load_image, rbd
 

 

class MultiImageMatcher:
    def __init__(self):
       
        """
        初始化多图像匹配器
         
        """
         
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          
 
        # --- 更换为 ALIKED 特征提取器 ---
        self.extractor = SuperPoint(
            max_num_keypoints=8192,          # 或 None(更慢更占显存)
            detection_threshold=0.0005       # 比 0.005 密很多
        ).eval().to(self.device)

        self.matcher = LightGlue(
            features="superpoint",
            depth_confidence=-1,             # 关 early stopping
            width_confidence=-1,             # 关 point pruning
            filter_threshold=0.10            # 想“更稳”就↑；想“更密”才↓
        ).eval().to(self.device)

    def load_images(self, image_paths):
        """
        加载图像列表
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            list: 加载的图像列表
        """
        images = []
        for path in image_paths:
            image = load_image(path)
            images.append(image)
        return images
    
    def match_images(self, images, mask=None):
        """
        将后续图像与第一张图像进行匹配
        
        Args:
            images: 图像列表，第一张图像为主视角
            
        Returns:
            list: 每个元素是一个元组 (matches, m_kpts0, m_kpts1)，表示与主图像的匹配结果
        """
        if len(images) < 2:
            raise ValueError("需要至少两张图像来进行匹配")
            
        # 第一张图像作为主视角
        main_image_tensor = images[0].to(self.device)
        main_feats = self.extractor.extract(main_image_tensor)

        # 新的mask使用方式：先提取所有特征点，然后根据mask进行筛选
        if mask is not None:
            # 获取特征点坐标 (N, 2)
            # 提取器返回的特征点形状为 (1, N, 2)，我们处理时去掉批次维度
            kpts = main_feats['keypoints'][0]
            # 将坐标转换为整数，用于在mask中索引
            kpts_int = torch.round(kpts).long()

            # 确保坐标在图像范围内
            h, w = mask.shape
            # mask的索引是 (y, x)，而kpts是 (x, y)
            keep = (kpts_int[..., 0] >= 0) & (kpts_int[..., 0] < w) & \
                   (kpts_int[..., 1] >= 0) & (kpts_int[..., 1] < h) & \
                   (torch.from_numpy(mask).to(kpts_int.device)[kpts_int[..., 1], kpts_int[..., 0]] > 0)

            # 筛选 main_feats 字典中的所有项
            for k in main_feats:
                # 只筛选与特征点数量相关的张量，跳过 'image_size' 等
                if main_feats[k].shape[1] == kpts.shape[0]:
                    main_feats[k] = main_feats[k][0][keep].unsqueeze(0)
        
        results = []
        
        # 对于后续的每张图像，与主图像进行匹配
        for i in range(1, len(images)):
            # 提取当前图像的特征
            curr_feats = self.extractor.extract(images[i].to(self.device))
            
            # 匹配主图像和当前图像的特征
            matches_result = self.matcher({"image0": main_feats, "image1": curr_feats})
            main_feats_copy, curr_feats_copy, matches_result = [rbd(x) for x in [main_feats, curr_feats, matches_result]]
            
            # 获取匹配的关键点
            main_kpts = main_feats_copy["keypoints"]
            curr_kpts = curr_feats_copy["keypoints"]
            matches = matches_result["matches"]
            m_kpts0 = main_kpts[matches[..., 0]]
            m_kpts1 = curr_kpts[matches[..., 1]]
            
            # 保存结果
            results.append((matches, m_kpts0, m_kpts1))
            
        return results
    
    def process_image_paths(self, image_paths, mask=None):
        """
        处理图像路径列表，加载图像并执行匹配
        
        Args:
            image_paths: 图像路径列表
            mask: 应用于第一张图像的运动遮罩 (Numpy array)
            
        Returns:
            tuple: (images, matches_results) 图像列表和匹配结果
        """
        # 加载所有图像
        images = self.load_images(image_paths)
        
        # 执行匹配
        matches_results = self.match_images(images, mask=mask)
        
        return images, matches_results

    def get_match_image(self, image0, image1, match_result, mask=None, movement_threshold=10.0):
        """
        为一对图像和它们的匹配结果生成一张可视化图像。
        动态点将用更大的红点和红线绘制。
        :param image0: PyTorch Tensor 格式的第一张图
        :param image1: PyTorch Tensor 格式的第二张图
        :param match_result: (matches, m_kpts0, m_kpts1) 的元组
        :param mask: 应用于主图像的运动遮罩 (Numpy array, HxW)
        :param movement_threshold: 判定为“动态”的像素位移阈值
        :return: OpenCV BGR 格式的可视化图像
        """
        _, m_kpts0, m_kpts1 = match_result

        # 将 PyTorch 的 RGB tensor 转换为 OpenCV 的 BGR numpy 数组
        img0_bgr = cv2.cvtColor((image0.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        img1_bgr = cv2.cvtColor((image1.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # 如果提供了mask，则在第一张图上可视化它
        if mask is not None:
            # 创建一个半透明的红色叠加层来表示被遮盖的区域
            overlay = img0_bgr.copy()
            # 将mask为0的区域（静态背景）涂成红色
            overlay[mask == 0] = [0, 0, 255] # BGR for red
            # 将原图和叠加层混合，使红色区域半透明
            img0_bgr = cv2.addWeighted(overlay, 0.5, img0_bgr, 0.5, 0)

        # 创建一个拼接后的大图用于绘制
        h0, w0 = img0_bgr.shape[:2]
        h1, w1 = img1_bgr.shape[:2]
        vis = np.zeros((max(h0, h1), w0 + w1, 3), np.uint8)
        vis[:h0, :w0] = img0_bgr
        vis[:h1, w0:] = img1_bgr

        # 定义动态和静态点的样式
        dynamic_color = (0, 0, 255)  # 红色 (BGR)
        static_color = (0, 255, 0)   # 绿色 (BGR)
        dynamic_radius = 5
        static_radius = 3

        # 手动绘制每个匹配，并根据位移区分颜色
        for p0, p1 in zip(m_kpts0.cpu().numpy(), m_kpts1.cpu().numpy()):
            p0_t = tuple(map(int, p0))
            p1_t = tuple(map(int, p1))
            
            # 计算位移
            distance = np.linalg.norm(p0 - p1)
            
            if distance > movement_threshold:
                color, radius = dynamic_color, dynamic_radius
            else:
                color, radius = static_color, static_radius

            # 绘制关键点和连线
            cv2.circle(vis, p0_t, radius, color, -1)
            cv2.circle(vis, (p1_t[0] + w0, p1_t[1]), radius, color, -1)
            cv2.line(vis, p0_t, (p1_t[0] + w0, p1_t[1]), color, 1)

        return vis
    
    def visualize_matches(self, images, matches_results, image_paths=None, save_dir=None):
        """
        使用OpenCV可视化匹配结果。
        
        Args:
            images: 图像列表
            matches_results: 匹配结果列表
            image_paths: 图像路径列表（可选）
            save_dir: 保存可视化结果的目录路径（可选）
        """
        num_matches = len(matches_results)
        main_image_np = (images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
        
        for i in range(num_matches):
            matches, m_kpts0, m_kpts1 = matches_results[i]
            print(f'第 {i+1} 张图像与主图像之间找到 {len(matches)} 个匹配点')
            match_img = self.get_match_image(images[0], images[i+1], matches_results[i])

            window_title = "Matches"
            if image_paths:
                window_title = f"{image_paths[0].name} vs {image_paths[i+1].name}"

            if save_dir:
                output_filename = save_path / f"match_{image_paths[0].stem}_vs_{image_paths[i+1].stem}.png"
                cv2.imwrite(str(output_filename), match_img)
                print(f"匹配结果已保存到: {output_filename}")

            # --- 调整图像大小以便于显示 ---
            max_display_width = 1920  # 设置显示窗口的最大宽度
            h, w = match_img.shape[:2]
            
            if w > max_display_width:
                scale = max_display_width / w
                new_w = int(w * scale)
                new_h = int(h * scale)
                display_img = cv2.resize(match_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                display_img = match_img

            cv2.imshow(window_title, display_img)
            key = cv2.waitKey(0) # 等待按键
            if key == ord('q') or key == 27: # 如果是'q'或'Esc'
                break # 退出循环
        cv2.destroyAllWindows()


# 原始示例代码的使用方式
if __name__ == "__main__":
    
    
    # 创建匹配器实例
    matcher = MultiImageMatcher()
    
    # 加载图像
    data_dir = Path(r"/home/crgj/wdd/data/sync/metashape/frame000026/test/")
    image_paths = sorted(list(data_dir.iterdir()))  # 加载所有图像
    
    if len(image_paths) < 2:
        print("需要至少两张图像来进行匹配")
        exit()
    
    # 处理图像
    images, matches_results = matcher.process_image_paths(image_paths)
    
    # 可视化结果（可选）
    # matcher.visualize_matches(images, matches_results, image_paths)
    # 将可视化结果保存到 'output' 文件夹
    matcher.visualize_matches(images, matches_results, image_paths, save_dir="output")
