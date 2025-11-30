

from utils.motion_utils import detect_motion 
from dataloader import dataloader, qvec2rotmat
from utils.web_logger import WebLogger
from utils.match_utils import MultiImageMatcher
from utils.calculation_spatial_distance import calculate_batch_spatial_distances

import os
import webbrowser
from pathlib import Path
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go

if __name__ == "__main__":
    
    # 1. 参数设置
    root_dir = "/home/crgj/wdd/data/sync/metashape/"  # 修改为你的数据根目录
    target_cam_name = "017"  # 指定相机序号
    worker_cam_name = "019"  # 需要被时间对齐的摄像机名字指定工作相机序号
    search_window_width = 15  # 定义搜索窗口的总宽度 (建议为奇数)
    closing_iters=2
    num_to_average =  30 



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
    motion_result = detect_motion(target_cam, thresh=20, closing_iters=closing_iters)

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
        if search_window_width % 2 == 0:
            print(f"Warning: search_window_width ({search_window_width}) is even. For a symmetric window, an odd number is recommended.")
        
        half_width = search_window_width // 2
        search_window = list(range(-half_width, half_width + 1))
        actual_search_window = [] # 存储实际使用的偏移量
        num_worker_frames = len(worker_cam.image_paths)
        
        for offset in search_window:
            worker_idx = max_idx + offset
            # 关键：检查索引是否越界
            if 0 <= worker_idx < num_worker_frames:
                actual_search_window.append(offset)
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

       
        # --- Helper function to get camera parameters ---
        def get_cam_params(loader, frame_idx):
            path = loader.image_paths[frame_idx]
            frame_name = Path(path).parent.parent.name
            
            if frame_name not in loader.frame_images or not loader.frame_images[frame_name]:
                return None, None, None, None

            # 修复：根据文件名找到正确的图像对象
            target_img_name = Path(path).name
            img_obj = None
            for obj in loader.frame_images[frame_name].values():
                if obj.name == target_img_name:
                    img_obj = obj
                    break
            
            if img_obj is None: # 如果在该帧中找不到对应的图像对象
                return None, None, None, None

            cam_obj = loader.cameras[img_obj.camera_id]

            K = np.array([[cam_obj.params[0], 0, cam_obj.params[2]],
                          [0, cam_obj.params[1], cam_obj.params[3]],
                          [0, 0, 1]])
            R = qvec2rotmat(img_obj.qvec)
            T = img_obj.tvec
            return K, R, T, img_obj

        # 获取 target_cam 的参数
        K1, R1, T1, img_obj1 = get_cam_params(target_cam, max_idx)

        # 检查是否成功获取参数
        if K1 is None:
            logger.add_text("Error: Could not retrieve camera parameters for the target frame. Skipping spatial distance calculation.", "Error")
        else:
            logger.add_text("Calculating spatial distance error for each match...", title="Step 7: Calculate Spatial Error")
 
        # --- 6. 将匹配结果写入日志，并找出最佳匹配 ---
        logger.add_text("Finding the best match based on minimum spatial distance error...", title="Step 6: Analyze Matching Results")
        
        match_errors = []
        # 1. 先计算所有匹配对的误差
        for i, result in enumerate(matches_results):
            _, m_kpts0, m_kpts1 = result
            num_matches = len(m_kpts0) if m_kpts0 is not None else 0
            
            error = float('inf') # 默认误差为无穷大
            if K1 is not None and num_matches > 0:
                worker_idx = max_idx + actual_search_window[i] # 使用 actual_search_window 保证索引正确
                K2, R2, T2, _ = get_cam_params(worker_cam, worker_idx)
                if K2 is not None:
                    distances = calculate_batch_spatial_distances(
                        m_kpts0.cpu().numpy(), K1, R1, T1,
                        m_kpts1.cpu().numpy(), K2, R2, T2
                    )
                    sorted_distances = np.sort(distances)[::-1]
                    
                    if len(sorted_distances) > num_to_average:
                        top_distances = sorted_distances[:num_to_average] 
                    else:
                        top_distances=np.inf
                    
                    error = np.mean(top_distances)
                    
            match_errors.append(error)

        #如果match_errors中的元素小于0.则将其设置为match_errors中最大元素的数值 
        match_errors_np = np.array(match_errors)
        if np.any(match_errors_np == np.inf) or np.any(match_errors_np < 0):
            valid_errors = match_errors_np[match_errors_np != float('inf')]
            max_valid_error = np.max(valid_errors[valid_errors >= 0]) if len(valid_errors[valid_errors >= 0]) > 0 else float('inf')
            match_errors = [max_valid_error if (error < 0 or error == float('inf')) else error for error in match_errors]

        # 2. 找到误差最小的索引作为最佳匹配
        if match_errors:
            best_match_idx = np.argmin(match_errors)
        else:
            best_match_idx = -1

        # --- (新增) 使用 pyecharts 绘制空间距离误差曲线图 ---
        if match_errors and best_match_idx != -1:
            # 过滤掉无穷大的误差值以便绘图
            plot_errors = [round(e, 4) if e != float('inf') else None for e in match_errors]
            best_offset = actual_search_window[best_match_idx]

            # 使用 Plotly 创建图表
            fig = go.Figure()

            # 添加误差曲线
            fig.add_trace(go.Scatter(
                x=actual_search_window,
                y=plot_errors,
                mode='lines+markers',
                name='Spatial Error'
            ))

            # 高亮最佳匹配点
            fig.add_trace(go.Scatter(
                x=[best_offset],
                y=[match_errors[best_match_idx]],
                mode='markers',
                marker=dict(color='red', size=12, symbol='star'),
                name=f'Best Match (Offset: {best_offset})'
            ))

            # 更新图表布局
            fig.update_layout(
                title_text="Spatial Error vs. Frame Offset",
                xaxis_title="Frame Offset",
                yaxis_title="Mean Error (Top 50)"
            )
            
            # 将图表转换为HTML div片段
            chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            logger.add_html_snippet(chart_html, title="Spatial Error Curve")

        # 3. 遍历所有匹配结果，生成文本和图像日志
        for i, result in enumerate(matches_results):
            matches, m_kpts0, m_kpts1 = result
            num_matches = len(matches)
            
            # 获取包含上一级目录的文件名
            target_image_name = f"{Path(image_paths[0]).parent.parent.name}/{Path(image_paths[0]).name}"
            worker_image_name = f"{Path(image_paths[i+1]).parent.parent.name}/{Path(image_paths[i+1]).name}"
            
            caption = f"'{target_image_name}' vs '{worker_image_name}': {num_matches} matches."

            # 使用预先计算好的误差
            error = match_errors[i]
            if error != float('inf'):
                caption += f" Mean Error (Top 50): {error:.4f}"

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
