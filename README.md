# Multi-Video Sync for 4D Gaussian Splatting

本项目用于为 **4D 高斯溅射 (4D Gaussian Splatting, 4DGS)** 等动态三维重建任务提供一个高精度的多视角视频同步方案。

多机位拍摄的启动时间往往存在轻微延迟。哪怕是几十毫秒的偏差，也会让同一时间戳下的不同视角记录到不同的场景状态，从而大幅降低 4D 重建质量。该项目通过计算机视觉分析自动估计各视频之间的时间偏移，生成同步后的帧映射关系或重新采样的视频序列。

## 核心思路

核心原则是**利用动态物体的瞬时状态作为时间指纹**：静态背景在不同时间几乎不可区分，但动态物体在每一帧的姿态和位置都独一无二。

1. **动态区域检测 (Motion Detection)**：选择一个基准视频，通过帧差、光流或背景建模锁定包含运动物体的区域。
2. **特征点提取与匹配 (Feature Extraction & Matching)**：在动态区域内提取 SIFT、ORB 或 SuperPoint 特征，并在其他视频的时间窗口内搜索匹配。
3. **基于对极几何的帧搜索 (Epipolar Geometry Search)**：结合相机内外参，计算匹配点对的重投影误差，衡量几何一致性。
4. **同步帧选择 (Best Match Selection)**：在窗口内遍历候选帧，选取重投影误差最小的帧作为同步结果。
5. **生成帧映射 (Sync Mapping)**：对基准视频的所有帧重复上述流程，得到完整的跨视频帧对应表。

## 项目结构

```
.
├── dataloader.py
├── motion_utils.py
├── README.md
└── requirements.txt
```

> 备注：根据需要可扩展 `configs/`、`src/`、`data/` 等目录来组织更完整的实验流程。

## 使用方法

1. **安装依赖**

    ```bash
    pip install -r requirements.txt
    ```

2. **准备数据**

    - 将多视角视频放入你的数据目录（示例：`data/your_scene/videos/`）。
    - 使用 COLMAP 等工具估计相机位姿，并导出为 `transforms.json` 或等价格式供同步算法使用。

3. **运行同步**

    ```bash
    python main.py --config configs/default.yaml --data_dir data/your_scene
    ```

## 未来工作

- [ ] 引入更高效的特征提取与匹配算法，加速处理流程。
- [ ] 优化时间窗口搜索策略，减少不必要的帧对评估。
- [ ] 支持自动化的相机参数估计或自标定流程。
- [ ] 将输出直接整理为 4DGS 及相关框架可用的数据格式。
