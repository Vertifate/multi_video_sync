import os
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
import struct
import random

# ==========================================
# Binary Reading Utilities (保持不变)
# ==========================================

class Camera:
    def __init__(self, camera_id, model, width, height, params):
        self.camera_id = camera_id
        self.model = model
        self.width = width
        self.height = height
        self.params = params

    def __str__(self):
        return f"Camera(camera_id={self.camera_id}, model={self.model}, width={self.width}, height={self.height})"


class Image:
    def __init__(self, image_id, qvec, tvec, camera_id, name, xys, point3D_ids):
        self.image_id = image_id
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name
        self.xys = xys
        self.point3D_ids = point3D_ids

    def __str__(self):
        return f"Image(image_id={self.image_id}, name={self.name}, camera_id={self.camera_id})"


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, 8*num_params, 
                                     "d"*num_params)
            cameras[camera_id] = Camera(camera_id, model_name, width, height, params)
    return cameras


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, 
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                image_id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

CAMERA_MODELS = [
    "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL",
    "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV",
    "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE"
]

CAMERA_MODEL_IDS = dict()
for i, model_name in enumerate(CAMERA_MODELS):
    CAMERA_MODEL_IDS[i] = type('obj', (object,), {
        'model_id': i,
        'model_name': model_name,
        'num_params': {
            "SIMPLE_PINHOLE": 3, "PINHOLE": 4, "SIMPLE_RADIAL": 4, "RADIAL": 5,
            "OPENCV": 8, "OPENCV_FISHEYE": 8, "FULL_OPENCV": 12, "FOV": 5,
            "SIMPLE_RADIAL_FISHEYE": 4, "RADIAL_FISHEYE": 5, "THIN_PRISM_FISHEYE": 12
        }[model_name]
    })()


# ==========================================
# Modified DataLoader
# ==========================================

class DataLoader:
    """
    加载指定相机序号(target_camera_id)对应的所有帧数据
    """
    
    def __init__(self, root_dir, target_camera_id):
        self.root_dir = Path(root_dir)
        self.target_camera_id = str(target_camera_id)  # 确保是字符串，例如 "001"
        
        # 存储相机参数（只存储目标相机对应的参数）
        self.cameras = {} 
        
        # 存储图像信息（按帧组织），只存储目标相机的Image对象
        self.frame_images = defaultdict(dict)  # frame_id -> {image_id: Image}
        
        # 存储图像路径列表
        self.image_paths = [] 
        
        self.num_frames = 0
        
    def load_colmap_data(self, frame_dir, frame_id):
        """
        加载指定帧目录下的COLMAP数据，并过滤出目标相机的数据
        """
        sparse_dir = frame_dir / "sparse" / "0"
        if not sparse_dir.exists():
            return

        # 1. 先读取 Images，找到目标图片对应的 colmap internal camera_id
        images_bin = sparse_dir / "images.bin"
        target_colmap_cam_id = None
        
        if images_bin.exists():
            # 读取所有图片数据 (COLMAP二进制格式难以只读一部分，必须解析结构)
            all_images = read_images_binary(images_bin)
            
            # 过滤：只保留文件名匹配 target_camera_id 的图片
            # 例如 target_id="001", 匹配 "001.jpg", "001.png" 等
            for img_id, img_obj in all_images.items():
                # 获取文件名（不含扩展名）
                stem = Path(img_obj.name).stem
                
                if stem == self.target_camera_id:
                    self.frame_images[frame_id][img_id] = img_obj
                    target_colmap_cam_id = img_obj.camera_id
                    break # 假设每一帧里该相机只有一张图
        
        # 2. 读取该帧中所有的相机参数
        cameras_bin = sparse_dir / "cameras.bin"
        if cameras_bin.exists():
            all_cameras_in_frame = read_cameras_binary(cameras_bin)
            self.cameras.update(all_cameras_in_frame) # 将新读取的相机参数合并到总字典中

    def load_sequence(self):
        """
        主加载循环：遍历所有帧，寻找目标相机的图片和参数
        """
        # 获取所有帧目录
        frame_dirs = sorted([d for d in self.root_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('frame')], 
                          key=lambda x: int(x.name.replace('frame', '')))
        
        self.num_frames = len(frame_dirs)
        print(f"Scanning {self.num_frames} frames for camera '{self.target_camera_id}'...")
        
        for frame_dir in frame_dirs:
            frame_id = frame_dir.name
            
            # 1. 总是尝试加载COLMAP数据
            self.load_colmap_data(frame_dir, frame_id)

            # 2. 检查物理图像文件是否存在，并构建 image_paths 列表
            images_dir = frame_dir / "images"
            if not images_dir.exists():
                continue
            
            # (这部分逻辑在您的上一个版本中已经修复，保持不变)
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                p = images_dir / (self.target_camera_id + ext)
                if p.exists():
                    self.image_paths.append(str(p))
                    break
        
        print(f"Done. Found {len(self.image_paths)} images and {len(self.cameras)} camera param sets.")

    def get_images(self):
        """
        读取所有查找到的图片到内存
        :return: list of numpy images
        """
        images = []
        for path in self.image_paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
        return images

    def get_camera_params(self):
        """
        返回加载的唯一相机参数对象
        注意：这个函数现在可能意义不大，因为loader会加载所有相机的参数。
        但为了向后兼容，我们保留它，它会返回找到的第一个相机参数。
        """
        if not self.cameras:
            return None
        return list(self.cameras.values())[0]

    def print_info(self):
        print("=" * 50)
        print(f"Target Camera: {self.target_camera_id}")
        print(f"Total Frames Found: {len(self.image_paths)}")
        cam = self.get_camera_params()
        if cam:
            print(f"Camera Model: {cam.model}")
            print(f"Resolution: {cam.width}x{cam.height}")
        else:
            print("Warning: No COLMAP camera parameters found for this ID.")
        print("=" * 50)


def dataloader(root_dir, target_camera_id):
    """
    创建并返回一个DataLoader实例，仅针对指定相机
    Args:
        root_dir: 数据根目录
        target_camera_id: 相机序号字符串 (e.g., "001", "002")
        
    Returns:
        DataLoader实例
    """
    loader = DataLoader(root_dir, target_camera_id)
    loader.load_sequence()
    return loader


# 使用示例
if __name__ == "__main__":
    # 示例：只加载相机 "001" 的数据
    target_id = "001"
    root_dir = "metashape" # 修改为实际路径
    
    if os.path.exists(root_dir):
        loader = dataloader(root_dir, target_id)
        
        loader.print_info()
        
        # 获取所有图片数据
        # images = loader.get_images()
        # print(f"Loaded {len(images)} raw images into memory.")
        
        # 获取相机参数
        # cam_params = loader.get_camera_params()
        # print(cam_params)
    else:
        print(f"Directory {root_dir} not found.")