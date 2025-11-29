import cv2
import numpy as np
import os
import sys
from pathlib import Path


#环境配置：添加上级目录到 sys.path 以导入 dataloader
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
 
from dataloader import dataloader 

# ==========================================
# 基础图像处理工具
# ==========================================

def compute_diff_gray(img1, img2):
    """返回灰度差分（单通道 uint8）"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return diff

def postprocess_mask(diff, thresh=20, open_kernel=(5,5), median_k=5, closing_kernel=(30,30), closing_iters=1):
    """从 diff 生成二值 mask 并做滤波/开运算/扩张"""
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    
    # 中值滤波去除椒盐噪声
    if median_k and median_k > 1:
        mask = cv2.medianBlur(mask, median_k)
    
    # 开运算去除噪点
    if open_kernel is not None:
        kernel = np.ones(open_kernel, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 闭运算（先膨胀再腐蚀），用于填充物体内部的空洞，并平滑边缘
    if closing_iters and closing_iters > 0:
        kernel = np.ones(closing_kernel, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=closing_iters)
        
    return mask

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_mask_motion_folder(img_path, mask):
    """
    将 mask 保存到 img_path 的同级目录下的 motion_mask/ 文件夹中
    结构示例:
       原图: .../frame00050/images/001.jpg
       Mask: .../frame00050/motion_mask/001.png
    """
    img_dir = os.path.dirname(img_path)              # e.g. .../frame00050/images
    parent_dir = os.path.dirname(img_dir)            # e.g. .../frame00050
    
    # 创建 motion_mask 文件夹
    mask_dir = os.path.join(parent_dir, "motion_mask")
    ensure_dir(mask_dir)
    
    # 保持文件名一致，但建议使用 png 以免压缩损失（虽然 jpg 也可以）
    base_name = os.path.basename(img_path)
    save_path = os.path.join(mask_dir, base_name)
    
    # 替换扩展名为 .png (可选，如果想保持 .jpg 可注释掉下面这行)
    save_path = os.path.splitext(save_path)[0] + ".png"
    
    # 确保 mask 为 uint8
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
        
    cv2.imwrite(save_path, mask)
    return save_path

# ==========================================
# 核心逻辑
# ==========================================

def detect_motion(loader, thresh=20, closing_iters=2):
    """
    处理加载器中的序列：
    1. 找出运动最大的一帧。
    2. 生成并保存该帧的 Mask。
    """
    
    # 1. 获取所有图片和对应路径
    # 注意：loader.get_images() 会将所有图片读入内存，如果序列特别长可能需要优化，
    # 但根据你的需求描述，这里直接读取。
    images = loader.get_images()
    paths = loader.image_paths
    
    num_images = len(images)
    if num_images < 2:
        print("Error: 图片数量少于2帧，无法计算运动。")
        return None

    print(f"Processing sequence: {num_images} frames loaded.")

    # 2. 遍历序列，计算相邻帧差分，找出运动最大的位置
    max_score = -1.0
    max_idx = -1
    
    # 从第1帧开始（与第0帧对比）
    for i in range(1, num_images):
        # 计算两帧之间的差异量
        diff = compute_diff_gray(images[i-1], images[i])
        score = np.sum(diff) # 简单的像素差分求和作为运动得分
        
        if score > max_score:
            max_score = score
            max_idx = i

    if max_idx == -1:
        print("Warning: 未检测到有效运动。")
        return None

    # 3. 获取最大运动帧的信息
    target_img = images[max_idx]
    prev_img = images[max_idx - 1]
    target_path = paths[max_idx]
    
    # 提取 frame 文件夹名称用于显示 (假设路径结构 .../frameXXXXX/images/001.jpg)
    try:
        frame_folder = Path(target_path).parent.parent.name
    except:
        frame_folder = "unknown_frame"

    # 4. 生成 Mask
    raw_diff = compute_diff_gray(prev_img, target_img)
    mask = postprocess_mask(raw_diff, thresh=thresh, closing_iters=closing_iters)
    
    # 5. 将结果打包成字典返回
    result = {
        "camera_id": loader.target_camera_id,
        "max_motion_index": max_idx,
        "frame_folder": frame_folder,
        "target_image_path": target_path,
        "motion_score": max_score,
        "mask": mask,
        "target_image": target_img,
        "prev_image": prev_img,
        "raw_diff": raw_diff
    }

    return result

# ==========================================
# Main
# ==========================================

def main():

    # 1. 参数设置
    root_dir = "/home/crgj/wdd/data/sync/metashape/"  # 修改为你的数据根目录
    target_cam = "001"  # 指定相机序号
    
    if not os.path.exists(root_dir):
        print(f"Error: 目录不存在 {root_dir}")
        return

    # 2. 初始化 DataLoader (使用新的 API)
    print(f"Initializing DataLoader for camera {target_cam}...")
    loader = dataloader(root_dir, target_cam)
    
    # 3. 执行处理
    detect_motion(loader, thresh=20, closing_iters=2)

if __name__ == "__main__":
    main()