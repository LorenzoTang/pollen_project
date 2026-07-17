import json
from pathlib import Path
from PIL import Image, ImageDraw
import random


# 数据路径
ROOT = Path(
    r"C:\Users\22577\Desktop\pollen_project\data\processed\pollen_detection"
)


split = "train"


# 读取 COCO 标注
json_path = (
    ROOT /
    "annotations" /
    f"{split}.json"
)


with open(
    json_path,
    encoding="utf-8"
) as f:
    data = json.load(f)


# 随机选择一张图片
img_info = random.choice(
    data["images"]
)


image_id = img_info["id"]


# 打开图片
img_path = (
    ROOT /
    split /
    "images" /
    img_info["file_name"]
)


img = Image.open(img_path)


draw = ImageDraw.Draw(img)


count = 0


# 找属于这张图片的bbox
for ann in data["annotations"]:

    if ann["image_id"] == image_id:

        x, y, w, h = ann["bbox"]


        draw.rectangle(
            [
                x,
                y,
                x + w,
                y + h
            ],
            outline="red",
            width=5
        )

        count += 1



print("================")
print("图片:", img_info["file_name"])
print("尺寸:", img.size)
print("目标数量:", count)



# 缩放显示
img.thumbnail(
    (1200, 900)
)


img.show()