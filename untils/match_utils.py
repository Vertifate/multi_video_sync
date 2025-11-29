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

from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd
 

 

class MultiImageMatcher:
    def __init__(self, device=None):
        """
        初始化多图像匹配器
        
        Args:
            device: 计算设备 ('cuda', 'cpu' 等)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # 加载特征提取器和匹配器
        self.extractor = SuperPoint(max_num_keypoints=2048000).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        
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
    
    def match_images(self, images):
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
        main_image = images[0]
        main_feats = self.extractor.extract(main_image.to(self.device))
        
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
    
    def process_image_paths(self, image_paths):
        """
        处理图像路径列表，加载图像并执行匹配
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            tuple: (images, matches_results) 图像列表和匹配结果
        """
        # 加载所有图像
        images = self.load_images(image_paths)
        
        # 执行匹配
        matches_results = self.match_images(images)
        
        return images, matches_results
    
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
            print(f'第{i+1}张图像与主图像之间找到 {len(matches)} 个匹配点')

            # 准备OpenCV需要的数据格式
            # 将 PyTorch 的 RGB tensor 转换为 OpenCV 的 BGR numpy 数组
            main_image_bgr = cv2.cvtColor((images[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            curr_image_bgr = cv2.cvtColor((images[i+1].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # 将torch tensor关键点转换为cv2.KeyPoint对象列表
            kpts0_cv = [cv2.KeyPoint(p[0].item(), p[1].item(), 1) for p in m_kpts0]
            kpts1_cv = [cv2.KeyPoint(p[0].item(), p[1].item(), 1) for p in m_kpts1]
            
            # 创建DMatch对象
            dmatches = [cv2.DMatch(_queryIdx=j, _trainIdx=j, _distance=0) for j in range(len(kpts0_cv))]
            
            # 使用cv2.drawMatches绘制匹配
            match_img = cv2.drawMatches(main_image_bgr, kpts0_cv, curr_image_bgr, kpts1_cv, dmatches, None, 
                                        matchColor=(0, 255, 0),  # BGR for lime green
                                        singlePointColor=None, 
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

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
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建匹配器实例
    matcher = MultiImageMatcher(device)
    
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
