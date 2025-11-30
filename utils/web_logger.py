import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

class WebLogger:
    """
    一个简单的HTML日志记录器，用于生成带有文本和图片的分步日志。
    """
    def __init__(self, log_dir="output", filename="log.html"):
        """
        初始化日志记录器。
        :param log_dir: 日志和图片保存的目录。
        :param filename: HTML日志文件的名称。
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.log_path = self.log_dir / filename
        self.step_count = 0

        # 初始化HTML文件
        self._write_header()

    def _write_header(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Process Log</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; color: #333; max-width: 960px; margin: 20px auto; padding: 0 20px; }}
        .log-entry {{ border-left: 3px solid #eee; padding-left: 20px; margin-bottom: 20px; }}
        .step {{ font-size: 1.5em; font-weight: bold; color: #4a4a4a; margin-bottom: 10px; border-bottom: 2px solid #eee; padding-bottom: 5px; }}
        .text {{ white-space: pre-wrap; background-color: #f9f9f9; border: 1px solid #ddd; padding: 10px; border-radius: 4px; font-family: 'Courier New', Courier, monospace; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; }}
        figcaption {{ font-style: italic; color: #777; text-align: center; margin-top: 5px; }}
        .image-grid {{ display: flex; flex-wrap: wrap; gap: 10px; justify-content: center; }}
        .image-grid figure {{ margin: 0; flex: 1 1 300px; }}
    </style>
</head>
<body>
<h1>Process Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h1>
""") # yapf: disable

    def add_text(self, text, title=None):
        """向日志中添加一个文本条目。"""
        self.step_count += 1
        title = title or f"Step {self.step_count}: Log Message"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f'<div class="log-entry">\n')
            f.write(f'  <div class="step">{title}</div>\n')
            f.write(f'  <div class="text">{text}</div>\n')
            f.write(f'</div>\n')

    def add_image(self, image_data, caption, filename=None):
        """
        向日志中添加一个图像条目。图像将被保存到日志目录。
        :param image_data: Numpy数组格式的图像 (来自OpenCV)。
        :param caption: 图像的标题。
        :param filename: 保存图像的文件名。如果为None，则自动生成。
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"image_{timestamp}.png"
        
        save_path = self.log_dir / filename
        cv2.imwrite(str(save_path), image_data)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f'<div class="log-entry">\n')
            f.write(f'  <figure>\n')
            f.write(f'    <img src="{filename}" alt="{caption}">\n')
            f.write(f'    <figcaption>{caption}</figcaption>\n')
            f.write(f'  </figure>\n')
            f.write(f'</div>\n')

    def add_image_grid(self, images_with_captions, title=None):
        """
        向日志中添加一个图像网格，用于并排显示多张图片。
        :param images_with_captions: 一个元组列表 [(image_data, caption, filename), ...]
        :param title: 网格的标题。
        """
        self.step_count += 1
        title = title or f"Step {self.step_count}: Image Comparison"

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f'<div class="log-entry">\n')
            f.write(f'  <div class="step">{title}</div>\n')
            f.write(f'  <div class="image-grid">\n')

            for image_data, caption, filename in images_with_captions:
                if filename is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"image_{timestamp}.png"
                
                save_path = self.log_dir / filename
                cv2.imwrite(str(save_path), image_data)

                f.write(f'    <figure>\n')
                f.write(f'      <img src="{filename}" alt="{caption}">\n')
                f.write(f'      <figcaption>{caption}</figcaption>\n')
                f.write(f'    </figure>\n')
            
            f.write(f'  </div>\n</div>\n')

    def add_html_snippet(self, html_content, title=None):
        """向日志中添加一个原始的HTML代码片段（例如图表）。"""
        self.step_count += 1
        title = title or f"Step {self.step_count}: Interactive Chart"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f'<div class="log-entry">\n')
            f.write(f'  <div class="step">{title}</div>\n')
            f.write(f'  {html_content}\n')
            f.write(f'</div>\n')

    def close(self):
        """关闭HTML标签。"""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("</body>\n</html>")