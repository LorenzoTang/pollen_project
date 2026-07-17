from pathlib import Path
import pandas as pd
import json
import shutil
import random
from tqdm import tqdm
from collections import defaultdict, Counter


# ==========================
# 路径
# ==========================

IMAGE_ROOT = Path(
    r"C:\Users\22577\Desktop\pollen_project\data\raw\训练数据\Raw Data\Images"
)

ANNOTATION_ROOT = Path(
    r"C:\Users\22577\Desktop\pollen_project\data\raw\训练数据\Raw Data\Annotations"
)

OUTPUT_ROOT = Path(
    r"C:\Users\22577\Desktop\pollen_project\data\processed\pollen_detection"
)


# ==========================
# 选择类别
# ==========================

SELECT_CLASSES = [
    "1.Thymbra",
    "2.Erica",
    "3.Castanea",
    "4.Eucalyptus",
    "5.Myrtus"
]


VAL_RATIO = 0.2


# ==========================
# 主程序
# ==========================

def convert():

    # 清理输出
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)


    (OUTPUT_ROOT / "annotations").mkdir(
        parents=True,
        exist_ok=True
    )


    # 类别id
    category_map = {}

    categories = []


    for idx, cls in enumerate(SELECT_CLASSES):

        category_map[cls] = idx

        categories.append(
            {
                "id": idx,
                "name": cls
            }
        )


    # --------------------------------
    # 读取所有图片和bbox
    # --------------------------------

    dataset = []


    print("读取CSV...")


    for cls in SELECT_CLASSES:


        csv_files = list(
            (ANNOTATION_ROOT / cls).glob("*.csv")
        )


        if not csv_files:
            print("缺少CSV:", cls)
            continue


        csv_path = csv_files[0]


        df = pd.read_csv(csv_path)


        print(
            cls,
            "images:",
            df["file"].nunique(),
            "bbox:",
            len(df)
        )


        # 按图片分组
        grouped = df.groupby("file")


        for filename, group in grouped:


            item = {

                "class": cls,

                "filename": filename,

                "width": int(
                    group.iloc[0]["resolution_x"]
                ),

                "height": int(
                    group.iloc[0]["resolution_y"]
                ),

                "boxes": []

            }


            for _, row in group.iterrows():

                item["boxes"].append(
                    [
                        int(row["x"]),
                        int(row["y"]),
                        int(row["w"]),
                        int(row["h"])
                    ]
                )


            dataset.append(item)


    print(
        "\n总图片:",
        len(dataset)
    )


    # --------------------------------
    # 划分 train val
    # --------------------------------

    random.seed(42)

    random.shuffle(dataset)


    split = int(
        len(dataset)
        *
        (1 - VAL_RATIO)
    )


    train_data = dataset[:split]

    val_data = dataset[split:]


    print(
        "train:",
        len(train_data)
    )

    print(
        "val:",
        len(val_data)
    )


    # --------------------------------
    # 生成COCO
    # --------------------------------

    def create_coco(data, split):


        img_dir = (
            OUTPUT_ROOT /
            split /
            "images"
        )


        img_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        coco_images = []

        coco_annotations = []


        image_id = 1

        ann_id = 1


        class_counter = Counter()


        for item in tqdm(
            data,
            desc=split
        ):


            cls = item["class"]

            filename = item["filename"]


            src = (
                IMAGE_ROOT /
                cls /
                filename
            )


            if not src.exists():

                print(
                    "图片不存在:",
                    src
                )

                continue


            # 防止同名
            new_filename = (
                cls +
                "_" +
                filename
            )


            dst = (
                img_dir /
                new_filename
            )


            shutil.copy2(
                src,
                dst
            )


            coco_images.append(
                {
                    "id": image_id,
                    "file_name": new_filename,
                    "width": item["width"],
                    "height": item["height"]
                }
            )


            for box in item["boxes"]:

                x,y,w,h = box


                coco_annotations.append(
                    {
                        "id": ann_id,

                        "image_id": image_id,

                        "category_id":
                            category_map[cls],

                        "bbox":
                            [
                                x,
                                y,
                                w,
                                h
                            ],

                        "area":
                            w*h,

                        "iscrowd":0
                    }
                )


                class_counter[cls]+=1


                ann_id += 1


            image_id += 1


        coco = {

            "images": coco_images,

            "annotations": coco_annotations,

            "categories": categories

        }


        with open(
            OUTPUT_ROOT /
            "annotations" /
            f"{split}.json",

            "w",

            encoding="utf-8"

        ) as f:


            json.dump(
                coco,
                f,
                indent=4
            )


        print(
            "\n",
            split,
            "类别统计:"
        )

        print(
            class_counter
        )


    create_coco(
        train_data,
        "train"
    )


    create_coco(
        val_data,
        "val"
    )


if __name__ == "__main__":

    convert()