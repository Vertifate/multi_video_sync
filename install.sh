#!/bin/bash

# 安装主项目依赖
echo "Installing main project dependencies..."
pip install -r requirements.txt

# 进入LightGlue目录并安装
echo "Installing LightGlue dependencies..."
cd LightGlue
python -m pip install -e .

# 返回上级目录
cd ..

# 验证安装
echo "Verifying installation..."
python -c "import cv2; import numpy; import matplotlib; import torch; import lightglue; print('All dependencies installed successfully!')"

echo "Installation completed!"