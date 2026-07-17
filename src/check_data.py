import json
from pathlib import Path
from collections import Counter


root = Path(
    r"data/processed/pollen_detection"
)


for split in ["train", "val"]:

    print("================")
    print(split)


    img_dir = (
        root /
        split /
        "images"
    )


    json_path = (
        root /
        "annotations" /
        f"{split}.json"
    )


    with open(
        json_path,
        encoding="utf-8"
    ) as f:

        data=json.load(f)



    print(
        "图片数量:",
        len(list(img_dir.glob("*")))
    )


    print(
        "json images:",
        len(data["images"])
    )


    print(
        "bbox数量:",
        len(data["annotations"])
    )


    counter=Counter()


    for ann in data["annotations"]:

        counter[
            ann["category_id"]
        ] += 1


    print(
        "类别分布:",
        counter
    )


    print(
        "类别名称:"
    )

    for c in data["categories"]:
        print(
            c["id"],
            c["name"]
        )