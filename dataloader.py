import os
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
import struct
import random


class Camera:
    """
    表示一个COLMAP相机模型
    """
    def __init__(self, camera_id, model, width, height, params):
        self.camera_id = camera_id
        self.model = model
        self.width = width
        self.height = height
        self.params = params

    def __str__(self):
        return f"Camera(camera_id={self.camera_id}, model={self.model}, width={self.width}, height={self.height})"


class Image:
    """
    表示一个COLMAP图像
    """
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
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of required elements.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    """
    读取COLMAP cameras.bin文件
    :param path_to_model_file: cameras.bin文件路径
    :return: 字典，key为camera_id，value为Camera对象
    """
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
    """
    读取COLMAP images.bin文件
    :param path_to_model_file: images.bin文件路径
    :return: 字典，key为image_id，value为Image对象
    """
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
    """
    将四元数转换为旋转矩阵
    :param qvec: 四元数 [w, x, y, z]
    :return: 3x3 旋转矩阵
    """
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


# COLMAP相机模型定义
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
            "SIMPLE_PINHOLE": 3,
            "PINHOLE": 4,
            "SIMPLE_RADIAL": 4,
            "RADIAL": 5,
            "OPENCV": 8,
            "OPENCV_FISHEYE": 8,
            "FULL_OPENCV": 12,
            "FOV": 5,
            "SIMPLE_RADIAL_FISHEYE": 4,
            "RADIAL_FISHEYE": 5,
            "THIN_PRISM_FISHEYE": 12
        }[model_name]
    })()


class DataLoader:
    """
    加载按帧组织的图像数据和COLMAP相机参数
    数据结构为:
    root/
    ├── frame000000/
    │   ├── images/
    │   │   ├── 000.jpg
    │   │   ├── 001.jpg
    │   │   └── ...
    │   ├── sparse/
    │   │   └── 0/
    │   │       ├── cameras.bin
    │   │       ├── images.bin
    │   │       └── points3D.bin
    ├── frame000001/
    │   ├── images/
    │   │   ├── 000.jpg
    │   │   ├── 001.jpg
    │   │   └── ...
    │   ├── sparse/
    │   │   └── 0/
    │   │       ├── cameras.bin
    │   │       ├── images.bin
    │   │       └── points3D.bin
    """
    
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        
        # 存储相机参数（只从一个帧读取）
        self.cameras = {}
        
        # 存储图像信息（按帧组织）
        self.frame_images = defaultdict(dict)  # frame_id -> {image_id: Image}
        
        # 存储图像路径，按相机组织
        self.image_sequences = defaultdict(list)
        
        # 存储帧数和相机数
        self.num_frames = 0
        self.num_cameras = 0
        
    def load_colmap_data(self, frame_dir, frame_id):
        """
        加载指定帧目录下的COLMAP格式的相机参数和图像位姿
        """
        sparse_dir = frame_dir / "sparse" / "0"
        
        # 加载相机参数（只从第一个帧加载）
        cameras_bin = sparse_dir / "cameras.bin"
        if cameras_bin.exists() and not self.cameras:
            print(f"Loading cameras from {frame_dir.name}")
            self.cameras = read_cameras_binary(cameras_bin)
            
        # 加载图像位姿
        images_bin = sparse_dir / "images.bin"
        if images_bin.exists():
            images = read_images_binary(images_bin)
            self.frame_images[frame_id] = images
    
    def load_image_sequences(self):
        """
        加载图像序列，按照相机名称组织
        """
        # 获取所有帧目录
        frame_dirs = sorted([d for d in self.root_dir.iterdir() 
                           if d.is_dir() and d.name.startswith('frame')], 
                          key=lambda x: int(x.name.replace('frame', '')))
        
        self.num_frames = len(frame_dirs)
        print(f"Found {self.num_frames} frame directories")
        
        # 遍历每个帧目录
        for frame_idx, frame_dir in enumerate(frame_dirs):
            frame_id = frame_dir.name
            print(f"Processing frame {frame_idx+1}/{self.num_frames}: {frame_id}")
            
            # 加载该帧的COLMAP数据
            self.load_colmap_data(frame_dir, frame_id)
            
            images_dir = frame_dir / "images"
            if not images_dir.exists():
                print(f"Warning: images directory not found in {frame_dir}")
                continue
                
            # 在images目录中查找所有图像文件
            image_files = sorted([f for f in images_dir.iterdir() 
                                if f.is_file() and f.suffix.lower() in ['.png', '.jpg', '.jpeg']], 
                               key=lambda x: int(x.stem))
            
            # 按照文件名(000, 001等)作为相机ID组织图像
            for img_file in image_files:
                camera_id = img_file.stem  # 例如 "000", "001" 等
                self.image_sequences[camera_id].append((frame_id, str(img_file)))
        
        self.num_cameras = len(self.image_sequences)
        print(f"Found {self.num_cameras} cameras")
    
    def get_camera_sequence(self, camera_id):
        """
        获取指定相机的图像序列
        """
        if camera_id not in self.image_sequences:
            raise ValueError(f"Camera '{camera_id}' not found in sequences")
            
        image_paths = self.image_sequences[camera_id]
        images = []
        
        for frame_id, path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Could not load image {path}")
                
        return images
    
    def get_all_sequences(self):
        """
        获取所有相机的图像序列
        返回字典: {camera_id: [images]}
        注意：这会加载所有图像到内存中，可能会消耗大量内存和时间
        """
        print("Loading all image sequences into memory (this may take a while)...")
        sequences = {}
        for camera_id in sorted(self.image_sequences.keys()):
            print(f"Loading camera {camera_id}...")
            sequences[camera_id] = self.get_camera_sequence(camera_id)
        return sequences
    
    def load_data(self):
        """
        加载所有数据
        """
        print("Loading image sequences and COLMAP data...")
        self.load_image_sequences()
        total_images = sum(len(images) for images in self.frame_images.values())
        print(f"Loaded {len(self.cameras)} cameras")
        print(f"Loaded {total_images} images across {len(self.frame_images)} frames")
        
        # 显示统计信息
        self.print_statistics()
        
        return self
    
    def print_statistics(self):
        """
        打印数据集统计信息
        """
        print("=" * 50)
        print("Dataset Statistics:")
        print(f"  Root Directory: {self.root_dir}")
        print(f"  Total Frames: {self.num_frames}")
        print(f"  Total Cameras: {self.num_cameras}")
        total_image_instances = sum(len(seq) for seq in self.image_sequences.values())
        print(f"  Total Image Instances: {total_image_instances}")
        print("  Cameras:")
        for camera_id in sorted(self.image_sequences.keys()):
            print(f"    Camera {camera_id}: {len(self.image_sequences[camera_id])} frames")
        print("=" * 50)
        
    def print_random_camera_info(self):
        """
        随机选择一个相机并打印其内外参数信息
        """
        if not self.cameras:
            print("No camera data loaded!")
            return
            
        if not self.frame_images:
            print("No image data loaded!")
            return
            
        # 随机选择一个相机
        camera_id = random.choice(list(self.cameras.keys()))
        camera = self.cameras[camera_id]
        
        print("=" * 50)
        print("Random Camera Information:")
        print(f"Camera ID: {camera.camera_id}")
        print(f"Model: {camera.model}")
        print(f"Width: {camera.width}")
        print(f"Height: {camera.height}")
        print(f"Parameters: {camera.params}")
        
        # 随机选择一个帧，然后选择该帧中使用这个相机的图像
        frame_ids = list(self.frame_images.keys())
        if frame_ids:
            random_frame_id = random.choice(frame_ids)
            images_in_frame = self.frame_images[random_frame_id]
            
            # 查找使用该相机的图像
            image_for_camera = None
            for image in images_in_frame.values():
                if image.camera_id == camera_id:
                    image_for_camera = image
                    break
                    
            if image_for_camera:
                # 将四元数转换为旋转矩阵
                rotation_matrix = qvec2rotmat(image_for_camera.qvec)
                
                print("\nAssociated Image Information:")
                print(f"Image ID: {image_for_camera.image_id}")
                print(f"Image Name: {image_for_camera.name}")
                print(f"Quaternion (qvec): {image_for_camera.qvec}")
                print(f"Translation (tvec): {image_for_camera.tvec}")
                print("Rotation Matrix:")
                print(rotation_matrix)
                print(f"Frame ID: {random_frame_id}")
            else:
                print(f"\nNo image found for camera {camera_id} in frame {random_frame_id}")
        print("=" * 50)


def dataloader(root_dir):
    """
    创建并返回一个DataLoader实例
    Args:
        root_dir: 数据根目录
        
    Returns:
        DataLoader实例
    """
    loader = DataLoader(root_dir)
    loader.load_data()
    return loader


# 使用示例
if __name__ == "__main__":
    # 示例用法 - 只显示统计信息，不加载图像
    loader = dataloader("metashape")
    print("\nData loader initialized successfully!")
    print("To load image sequences, call loader.get_all_sequences() or loader.get_camera_sequence(camera_id)")
    
    # 随机读取并打印某个相机的内外参数
    loader.print_random_camera_info()
# 方便的 main 示例
def main():
    from dataloader import dataloader
    root_dir = "metashape"   # 修改为你的数据根目录
    loader = dataloader(root_dir)
    process_all_cameras(loader, group_size=30, thresh=20, expand_iters=4, reference_camera=None)

if __name__ == "__main__":
    main()
