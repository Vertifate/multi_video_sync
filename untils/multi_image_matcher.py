import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 添加LightGlue到路径
import sys
sys.path.append('LightGlue')

from LightGlue.lightglue import LightGlue, SuperPoint, DISK
from LightGlue.lightglue.utils import load_image, rbd
from LightGlue.lightglue import viz2d


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
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
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
    
    def visualize_matches(self, images, matches_results, image_paths=None):
        """
        可视化匹配结果
        
        Args:
            images: 图像列表
            matches_results: 匹配结果列表
            image_paths: 图像路径列表（可选）
        """
        num_matches = len(matches_results)
        
        for i in range(num_matches):
            # 创建子图
            axes = viz2d.plot_images([images[0], images[i+1]])
            
            # 绘制匹配点
            _, m_kpts0, m_kpts1 = matches_results[i]
            viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
            
            # 添加文本标签
            if image_paths:
                viz2d.add_text(0, f'{image_paths[0].name}')
                viz2d.add_text(1, f'{image_paths[i+1].name}')
                
            print(f'找到 {len(m_kpts0)} 个匹配点')
            
        plt.show()


# 原始示例代码的使用方式
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建匹配器实例
    matcher = MultiImageMatcher(device)
    
    # 加载图像
    data_dir = Path(r"D:\code\multi_video_sync\data\test")
    image_paths = sorted(list(data_dir.iterdir()))  # 加载所有图像
    
    if len(image_paths) < 2:
        print("需要至少两张图像来进行匹配")
        exit()
    
    # 处理图像
    images, matches_results = matcher.process_image_paths(image_paths)
    
    # 输出结果信息
    for i, (matches, m_kpts0, m_kpts1) in enumerate(matches_results):
        print(f'第{i+1}张图像与主图像之间找到 {len(matches)} 个匹配点')
    
    # 可视化结果（可选）
    matcher.visualize_matches(images, matches_results, image_paths)