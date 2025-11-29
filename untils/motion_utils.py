import cv2
import numpy as np
import os

def split_into_groups(seq, group_size=30):
    """按固定大小切组（seq 为 [(frame_id,path), ...]），最后一组不足也保留"""
    groups = []
    for i in range(0, len(seq), group_size):
        groups.append(seq[i : i + group_size])
    return groups


def compute_diff_gray(img1, img2):
    """返回灰度差分（单通道 uint8）和差分原图"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return diff


def postprocess_mask(diff, thresh=20, open_kernel=(5,5), median_k=5, expand_kernel=(16,16), expand_iters=5):
    """从 diff 生成二值 mask 并做滤波/开运算/扩张"""
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    if median_k and median_k > 1:
        mask = cv2.medianBlur(mask, median_k)
    if open_kernel is not None:
        kernel = np.ones(open_kernel, np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if expand_iters and expand_iters > 0:
        ek = np.ones(expand_kernel, np.uint8)
        mask = cv2.dilate(mask, ek, iterations=expand_iters)
    return mask


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_mask_in_parent_mask_folder(img_path, mask):
    """
    将 mask 保存到 img_path 的上一级目录下的 mask/ 文件夹中
    文件名与原图相同（例如 000.jpg -> ../mask/000.jpg）
    返回保存路径
    """
    img_dir = os.path.dirname(img_path)              # e.g. .../frame0000/images
    parent_dir = os.path.dirname(img_dir)            # e.g. .../frame0000
    mask_dir = os.path.join(parent_dir, "mask")
    ensure_dir(mask_dir)
    base_name = os.path.basename(img_path)           # 000.jpg
    save_path = os.path.join(mask_dir, base_name)
    # 确保 mask 为 uint8 单通道或三通道
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    cv2.imwrite(save_path, mask)
    return save_path


def detect_motion_index_for_group(images):
    """
    在单个相机组中找到 motion 最强的帧索引（相对于组起点的 0-based）
    images: list of BGR images, 长度可能 < 2
    返回：index（int），若组 < 2 则返回 0
    """
    n = len(images)
    if n < 2:
        return 0
    scores = []
    for i in range(1, n):
        diff = compute_diff_gray(images[i-1], images[i])
        scores.append(diff.sum())
    max_idx = int(np.argmax(scores)) + 1
    return max_idx


def process_all_cameras(loader, group_size=30, thresh=20, expand_iters=4, reference_camera=None):
    """
    主流程（稳健）：
      - 用 reference_camera（默认第一个）将时间序列按 group_size 切组
      - 在 reference_camera 的每组里找到 motion 最强的帧索引 (相对组内)
      - 对每个相机在同一组内用该索引对应的帧，与它自己的前一帧做差分 -> 得到独立 mask
      - mask 保存到该图像上一级目录的 mask/ 中，文件名与图像同名
    """
    camera_ids = sorted(loader.image_sequences.keys())
    if not camera_ids:
        raise ValueError("没有找到任何相机序列")

    # 选择参考相机（默认第一个）
    if reference_camera is None:
        reference_camera = camera_ids[0]
    if reference_camera not in camera_ids:
        raise ValueError(f"reference_camera '{reference_camera}' 不存在")

    ref_seq = loader.image_sequences[reference_camera]  # [(frame_id, path), ...]
    groups = split_into_groups(ref_seq, group_size)

    # 预构建每个相机的 frame_id -> index 映射，便于快速查找
    id_index_maps = {}
    for cam in camera_ids:
        map_d = {}
        for idx, (fid, path) in enumerate(loader.image_sequences[cam]):
            map_d[fid] = idx
        id_index_maps[cam] = map_d

    for g_idx, group in enumerate(groups):
        print(f"\n=== Processing group {g_idx} (size={len(group)}) ===")

        # 加载参考相机图像（本组）
        ref_images = []
        for fid, p in group:
            img = cv2.imread(p)
            if img is None:
                print(f"Warning: 无法读取参考相机图像 {p}，用黑图占位")
                # 用前一个图像大小或默认大小
                if ref_images:
                    h, w = ref_images[0].shape[:2]
                else:
                    h, w = 480, 640
                img = np.zeros((h, w, 3), dtype=np.uint8)
            ref_images.append(img)

        # 在参考相机中找 motion 最大的索引（相对组内）
        rel_max_idx = detect_motion_index_for_group(ref_images)
        # 为对齐方便，记录该索引对应的 frame_id（若该索引超出 group 长度则设为 None）
        if rel_max_idx < len(group):
            target_frame_id = group[rel_max_idx][0]
        else:
            target_frame_id = None

        print(f"Reference camera '{reference_camera}' group {g_idx} -> strongest motion at relative index {rel_max_idx}, frame_id={target_frame_id}")

        # 对每个相机独立计算 mask 并保存
        group_frame_ids = set([fid for fid, _ in group])
        for cam in camera_ids:
            seq = loader.image_sequences[cam]    # list of (fid,path)
            id_map = id_index_maps[cam]

            # 先尝试用 frame_id 对齐
            found = False
            if target_frame_id is not None and target_frame_id in id_map:
                global_idx = id_map[target_frame_id]
                found = True
            else:
                # 回退：使用相对全局索引映射（可能越界）
                global_start_idx = g_idx * group_size
                global_idx = global_start_idx + rel_max_idx
                if global_idx >= len(seq):
                    found = False
                else:
                    found = True

            if not found:
                print(f"  Camera {cam}: 未找到对应帧（group {g_idx}, rel_idx {rel_max_idx}），跳过")
                continue

            curr_fid, curr_path = seq[global_idx]

            # 尝试找到前一帧（同一相机）
            prev_idx = global_idx - 1
            # 我们希望前一帧也是在同一组内（可选），但更宽松的策略：只要存在就用
            if prev_idx >= 0:
                prev_fid, prev_path = seq[prev_idx]
                prev_img = cv2.imread(prev_path)
            else:
                prev_img = None

            curr_img = cv2.imread(curr_path)
            if curr_img is None:
                print(f"  Camera {cam}: 无法读取当前图像 {curr_path}，跳过")
                continue

            if prev_img is None:
                # 无法做差分 -> 生成全零 mask（尺寸同 curr）
                h, w = curr_img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                print(f"  Camera {cam}: 没有前一帧，生成全零 mask -> {curr_path}")
            else:
                diff = compute_diff_gray(prev_img, curr_img)
                mask = postprocess_mask(diff, thresh=thresh, expand_iters=expand_iters)
                print(f"  Camera {cam}: 计算 mask (prev: {prev_path} -> curr: {curr_path})")

            # 保存到上一级的 mask/ 文件夹，文件名与图像同名
            save_path = save_path = save_mask_in_parent_mask_folder(curr_path, mask)
            print(f"    saved -> {save_path}")

    print("\n==== All groups processed ====")


# 方便的 main 示例
def main():
    # 动态处理路径，以便脚本可以直接运行
    import sys
    from pathlib import Path
    # 将项目根目录（multi_video_sync）添加到 sys.path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    
    from dataloader import dataloader
    root_dir = "/home/crgj/wdd/data/sync/metashape/"   # 修改为你的数据根目录
    loader = dataloader(root_dir)
    process_all_cameras(loader, group_size=30, thresh=20, expand_iters=4, reference_camera=None)

if __name__ == "__main__":
    main()
