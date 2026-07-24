# -*- coding: utf-8 -*-
"""将 COCO 数据集中的中文图片文件名和类别名改为英文，解决 Windows 下 pycocotools GBK 解码错误。"""
import json
import os
import shutil

BASE = os.path.dirname(os.path.abspath(__file__))
SPLITS = ["train", "valid", "test"]

# 类别中文 -> 英文(拼音)映射
CAT_MAP = {
    "夹竹桃": "jiazhutao",  # 夹竹桃
    "构": "gou",                    # 构
    "楝": "lian",                   # 楝
    "樟": "zhang",                  # 樟
    "白玉兰": "baiyulan",   # 白玉兰
    "黑松": "heisong",          # 黑松
}


def process(split):
    d = os.path.join(BASE, split)
    ann_path = os.path.join(d, "_annotations.coco.json")

    # 备份 JSON
    bak = ann_path + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(ann_path, bak)

    with open(ann_path, encoding="utf-8") as f:
        data = json.load(f)

    # 1) 重命名图片文件 + 更新 file_name
    for i, img in enumerate(data["images"], start=1):
        old_name = img["file_name"]
        ext = os.path.splitext(old_name)[1].lower() or ".jpg"
        new_name = f"{split}_{i:04d}{ext}"

        old_path = os.path.join(d, old_name)
        new_path = os.path.join(d, new_name)
        if os.path.exists(old_path) and old_name != new_name:
            os.rename(old_path, new_path)
        img["file_name"] = new_name

    # 2) 更新类别名
    for c in data["categories"]:
        if c["name"] in CAT_MAP:
            c["name"] = CAT_MAP[c["name"]]

    # 3) 写回（保持 utf-8，不转义，仍是纯 ASCII 内容了）
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)

    print(f"[{split}] {len(data['images'])} images renamed, "
          f"cats: {[c['name'] for c in data['categories']]}")


if __name__ == "__main__":
    for s in SPLITS:
        process(s)
    print("Done. Backups saved as *.bak")
