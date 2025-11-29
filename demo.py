

from utils.motion_utils import detect_motion 
from dataloader import dataloader 
from utils.web_logger import WebLogger
from utils.match_utils import MultiImageMatcher

import os
import webbrowser
from pathlib import Path

if __name__ == "__main__":
    
    # 1. 参数设置
    root_dir = "/home/crgj/wdd/data/sync/metashape/"  # 修改为你的数据根目录
    target_cam_name = "001"  # 指定相机序号
    worker_cam_name = "065"  # 需要被时间对齐的摄像机名字指定工作相机序号



    output_dir = "output"

    if not os.path.exists(root_dir):
        print(f"Error: 目录不存在 {root_dir}")
        exit()

    # 初始化网页日志记录器
    logger = WebLogger(log_dir=output_dir, filename=f"log_cam_{target_cam_name}.html")

    # 2. 初始化 DataLoader (使用新的 API)
    logger.add_text(f"Initializing DataLoader for camera '{target_cam_name}' in root '{root_dir}'...", title="Step 1: Initialize DataLoader")
    target_cam = dataloader(root_dir, target_cam_name)
    target_cam.print_info() # 打印一些基本信息

    worker_cam = dataloader(root_dir, worker_cam_name)
    worker_cam.print_info() # 打印一些基本信息


    
    # 3. 首先寻找运动最大的一帧图像
    logger.add_text("Starting motion detection process...", title="Step 2: Detect Motion")
    motion_result = detect_motion(target_cam, thresh=20, closing_iters=15)

    if motion_result:
        # 记录找到的关键帧信息
        info_text = (
            f"Camera ID: {motion_result['camera_id']}\n"
            f"Max Motion Frame Index: {motion_result['max_motion_index']}\n"
            f"Frame Folder: {motion_result['frame_folder']}\n"
            f"Image Path: {motion_result['target_image_path']}\n"
            f"Motion Score: {motion_result['motion_score']}"
        )
        logger.add_text(info_text, title="Step 3: Motion Detection Result")

        # 记录并保存生成的mask图像
        logger.add_image(motion_result['mask'], caption=f"Final Motion Mask for camera {target_cam_name}", filename=f"mask_cam_{target_cam_name}.png")
        print(f"Motion detection complete. Log saved to {logger.log_path.resolve()}")
     


        # --- 4. 准备用于特征匹配的图像路径列表 ---
        logger.add_text("Preparing image list for feature matching...", title="Step 4: Prepare Image List")
        
        max_idx = motion_result['max_motion_index']
        
        # 目标图像路径总是第一个
        image_paths = [target_cam.image_paths[max_idx]]
        
        # 在 worker_cam 中以 max_idx 为中心，构建一个搜索窗口 (例如 -2 到 +2)
        search_window = [-3,-2, -1, 0, 1, 2,3]
        num_worker_frames = len(worker_cam.image_paths)
        
        for offset in search_window:
            worker_idx = max_idx + offset
            # 关键：检查索引是否越界
            if 0 <= worker_idx < num_worker_frames:
                image_paths.append(worker_cam.image_paths[worker_idx])

        # 修改日志文本，显示上一级目录
        log_text = "Images to be matched:\n" + "\n".join(
            [f"- {Path(p).parent.parent.name}/{Path(p).name}" for p in image_paths]
        )
        logger.add_text(log_text)

        # --- 5. 执行特征匹配 ---
        logger.add_text("Performing feature matching with LightGlue...", title="Step 5: Feature Matching")
        matcher = MultiImageMatcher()
        images, matches_results = matcher.process_image_paths(image_paths, mask=motion_result['mask'])

       

        # --- 6. 将匹配结果写入日志，并找出最佳匹配 ---
        logger.add_text("Finding the best match based on the number of feature points...", title="Step 6: Analyze Matching Results")
        
        best_match_idx = -1
        max_matches = -1

        # 1. 先找到最佳匹配的索引
        for i, result in enumerate(matches_results):
            matches, _, _ = result
            num_matches = len(matches)
            if num_matches > max_matches:
                max_matches = num_matches
                best_match_idx = i

        # 2. 遍历所有匹配结果，生成文本和图像日志
        for i, result in enumerate(matches_results):
            matches, _, _ = result
            num_matches = len(matches)
            
            # 获取包含上一级目录的文件名
            target_image_name = f"{Path(image_paths[0]).parent.parent.name}/{Path(image_paths[0]).name}"
            worker_image_name = f"{Path(image_paths[i+1]).parent.parent.name}/{Path(image_paths[i+1]).name}"
            
            caption = f"'{target_image_name}' vs '{worker_image_name}': {num_matches} matches"
            if i == best_match_idx:
                caption += " (Best Match)"

            match_image = matcher.get_match_image(images[0], images[i + 1], result, mask=motion_result['mask'])
            filename = f"match_{i}_{target_cam_name}_vs_{worker_cam_name}.png"
            logger.add_image(match_image, caption=caption, filename=filename)

    else:
        logger.add_text("Motion detection failed. Not enough images or no motion detected.", title="Error")
        print("Motion detection failed.")

    logger.close()

    # 自动在浏览器中打开日志文件
    log_file_path = logger.log_path.resolve()
    print(f"Opening log file in browser: {log_file_path}")
    # 使用 as_uri() 来确保跨平台的兼容性
    webbrowser.open(log_file_path.as_uri())
