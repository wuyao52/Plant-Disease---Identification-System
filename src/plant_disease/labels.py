from __future__ import annotations

import json
from pathlib import Path


PLANTVILLAGE_ZH: dict[str, str] = {
    "Apple___Apple_scab": "苹果黑星病",
    "Apple___Black_rot": "苹果黑腐病",
    "Apple___Cedar_apple_rust": "苹果雪松锈病",
    "Apple___healthy": "苹果健康",
    "Blueberry___healthy": "蓝莓健康",
    "Cherry_(including_sour)___Powdery_mildew": "樱桃白粉病",
    "Cherry_(including_sour)___healthy": "樱桃健康",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "玉米灰斑病",
    "Corn_(maize)___Common_rust_": "玉米普通锈病",
    "Corn_(maize)___Northern_Leaf_Blight": "玉米北方叶枯病",
    "Corn_(maize)___healthy": "玉米健康",
    "Grape___Black_rot": "葡萄黑腐病",
    "Grape___Esca_(Black_Measles)": "葡萄黑麻疹病",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "葡萄叶枯病",
    "Grape___healthy": "葡萄健康",
    "Orange___Haunglongbing_(Citrus_greening)": "柑橘黄龙病",
    "Peach___Bacterial_spot": "桃细菌性斑点病",
    "Peach___healthy": "桃健康",
    "Pepper,_bell___Bacterial_spot": "甜椒细菌性斑点病",
    "Pepper,_bell___healthy": "甜椒健康",
    "Potato___Early_blight": "马铃薯早疫病",
    "Potato___Late_blight": "马铃薯晚疫病",
    "Potato___healthy": "马铃薯健康",
    "Raspberry___healthy": "树莓健康",
    "Soybean___healthy": "大豆健康",
    "Squash___Powdery_mildew": "南瓜白粉病",
    "Strawberry___Leaf_scorch": "草莓叶焦病",
    "Strawberry___healthy": "草莓健康",
    "Tomato___Bacterial_spot": "番茄细菌性斑点病",
    "Tomato___Early_blight": "番茄早疫病",
    "Tomato___Late_blight": "番茄晚疫病",
    "Tomato___Leaf_Mold": "番茄叶霉病",
    "Tomato___Septoria_leaf_spot": "番茄斑枯病",
    "Tomato___Spider_mites Two-spotted_spider_mite": "番茄二斑叶螨",
    "Tomato___Target_Spot": "番茄靶斑病",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "番茄黄化曲叶病毒病",
    "Tomato___Tomato_mosaic_virus": "番茄花叶病毒病",
    "Tomato___healthy": "番茄健康",
}


def normalize_label(label: str) -> str:
    return label.replace("___", " - ").replace("_", " ").strip()


def display_label(label: str) -> str:
    chinese = PLANTVILLAGE_ZH.get(label)
    readable = normalize_label(label)
    return f"{chinese} ({readable})" if chinese else readable


def save_class_names(class_names: list[str], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, ensure_ascii=False, indent=2)


def load_class_names(path: str | Path) -> list[str]:
    with Path(path).open("r", encoding="utf-8") as file:
        values = json.load(file)
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ValueError("class name file must contain a JSON string list")
    return values
