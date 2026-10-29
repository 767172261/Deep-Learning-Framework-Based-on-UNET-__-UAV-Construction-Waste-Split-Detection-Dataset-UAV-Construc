
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a2c8c4d1182f457a945e913f090d262c.jpeg)

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/79375d4b6bcc4216a01b7e2c7c166829.jpeg)

### 🏗️ 数据集类别（共 10 类）

| **类别英文名** | **中文名称** | **说明** |
|----------------|--------------|----------|
| `Debris` | 建筑垃圾 / 碎屑 | 建筑物倒塌后的瓦砾、混凝土块、钢筋等残骸 |
| `Water` | 水体 | 河流、湖泊、积水区域等 |
| `Building_No_Damage` | 无损建筑 | 完整未受损的建筑物 |
| `Building_Minor_Damage` | 轻微损坏建筑 | 局部墙体开裂、屋顶破损但主体结构完好 |
| `Building_Major_Damage` | 严重损坏建筑 | 大面积坍塌、结构倾斜但尚未完全倒塌 |
| `Building_Total_Destruction` | 全部损毁建筑 | 完全倒塌，仅剩废墟 |
| `Vehicle` | 车辆 | 各类汽车、卡车、工程车等 |
| `Road` | 道路 | 普通道路、公路、桥梁等交通设施 |
| `Tree` | 树木 | 绿化植被、林地、树木覆盖区 |
| `Pool` | 游泳池 | 住宅区或公共区域的泳池（可能为灾害评估中的特殊目标） |
| `Sand` | 沙地 | 沙滩、沙土区域，常见于沿海或沙漠地区 |

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9ad3c8ddb611456a9929a23395d893e2.jpeg)

### 🔍 应用场景

该数据集主要用于：
- **灾后评估**：地震、台风、洪水等自然灾害后的建筑物损毁程度分析
- **城市更新监测**：识别拆迁现场的建筑垃圾分布
- **遥感影像分割**：基于无人机或卫星图像进行语义分割（如 U-Net、DeepLabV3+）
- **AI 辅助救援决策**：快速定位可通行道路、被困区域和危险建筑

> 💡 注：`Pool` 和 `Sand` 可能是为特定地理环境（如沿海城市）设计的补充类别，有助于提升模型在复杂背景下的泛化能力。

---

### ✅ 总结

这是一个典型的 **高分辨率航拍/遥感图像语义分割数据集**，专注于 **灾害响应与城市基础设施状态评估**。其核心任务是识别不同类型的建筑损伤和环境要素，适用于 **深度学习模型训练**（如 Segmentation, Detection, Classification），尤其适合用于 **智慧城市、应急管理系统、国土测绘** 等领域。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/05a9fbce17f84493a99f9b4a7d26676a.jpeg)
基于 **无人机建筑垃圾分割检测数据集** 的完整实现方案，包含：

✅ **UNet 语义分割训练代码（PyTorch）**  
✅ **数据预处理脚本（JSON → Mask）**  
✅ **Web 系统搭建（Django + WebGIS）**  
✅ **模型推理与可视化**  
✅ **系统部署建议**

---

## 📁 一、数据集概览

| 项目 | 说明 |
|------|------|
| **数据集名称** | 无人机建筑垃圾分割检测数据集 |
| **采集设备** | DJI Mavic 2 |
| **图像数量** | 4,494 张 |
| **分辨率** | 4000×3000 |
| **采集高度** | 60m ~ 120m |
| **视角角度** | 垂直俯视（90°） |
| **标注格式** | JSON（Polygon）、Mask（PNG） |
| **总大小** | 22.1 GB |
| **任务类型** | 语义分割（Semantic Segmentation） |
| **核心类别** | `Debris`（建筑垃圾） |
| **其他类别** | Water, Building_No_Damage, Minor/Major/Total Damage, Vehicle, Road, Tree, Pool, Sand |

> 💡 虽然你提到“类别：建筑垃圾”，但根据 Recall-Confidence 曲线图，实际是 **多类语义分割数据集**，其中 `Debris` 是关键目标。

---

## 🗂️ 二、博主建议目录结构

```
debris_segmentation_system/
├── data/
│   ├── images/                  # 原始图像
│   ├── masks/                   # 分割掩码（PNG）
│   └── annotations/             # JSON 标注（可选）
├── models/
│   └── unet_debris.pth          # 训练好的 UNet 模型
├── train.py                     # UNet 训练脚本
├── infer.py                     # 推理脚本
├── web_app/
│   ├── views.py
│   ├── models.py
│   └── templates/
│       └── index.html
└── requirements.txt
```

---

## 📦 三、`requirements.txt`

```txt
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.0
numpy==1.24.3
Pillow==9.5.0
matplotlib==3.6.3
django==4.2.7
pyecharts==2.0.0
shapely==2.0.0
```

安装：
```bash
pip install -r requirements.txt
```

---

## 🔁 四、JSON → PNG 掩码转换（`utils/json_to_mask.py`）

```python
# utils/json_to_mask.py
import json
import cv2
import numpy as np
from pathlib import Path

def json_to_mask(json_path, output_dir, class_name="Debris"):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if 'annotations' not in data:
        return None

    h, w = data['imageHeight'], data['imageWidth']
    mask = np.zeros((h, w), dtype=np.uint8)

    for ann in data['annotations']:
        if ann['label'] == class_name and 'polygon' in ann:
            points = np.array(ann['polygon'], dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [points], color=255)

    output_path = Path(output_dir) / f"{Path(json_path).stem}.png"
    cv2.imwrite(str(output_path), mask)
    return output_path

if __name__ == "__main__":
    json_dir = "data/annotations"
    mask_dir = "data/masks"
    Path(mask_dir).mkdir(parents=True, exist_ok=True)

    for json_file in Path(json_dir).glob("*.json"):
        print(f"Processing {json_file.name}")
        json_to_mask(json_file, mask_dir)
    print("✅ 所有 JSON 已转为掩码！")
```

> 💡 假设 JSON 中的 `label` 字段为 `Debris`，且包含 `polygon` 数组。

---

## 🧠 五、UNet 模型定义（`models/unet.py`）

```python
# models/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownSample(64, 128)
        self.down2 = DownSample(128, 256)
        self.down3 = DownSample(256, 512)
        self.down4 = DownSample(512, 512)
        self.up1 = UpSample(1024, 256)
        self.up2 = UpSample(512, 128)
        self.up3 = UpSample(256, 64)
        self.up4 = UpSample(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.sigmoid(logits)
```

---

## 🚀 六、`train.py` —— UNet 训练脚本

```python
# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

class DebrisDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = torch.where(mask > 0, torch.tensor(1), torch.tensor(0)).float()

        return image, mask

# 数据增强
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = DebrisDataset("data/images", "data/masks", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

# 保存模型
torch.save(model.state_dict(), "models/unet_debris.pth")
print("✅ 模型已保存！")
```

---

## 🔍 七、`infer.py` —— 推理脚本

```python
# infer.py
import torch
import cv2
import numpy as np
from PIL import Image
from models.unet import UNet

model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load("models/unet_debris.pth"))
model.eval()

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = output.squeeze().cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    return pred_mask

# 使用示例
mask = predict_image("data/images/test.jpg")
cv2.imwrite("output_mask.png", mask)
```

---

## 🌐 八、Django Web 系统（`web_app/views.py`）

```python
from django.shortcuts import render
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage
from .models import DetectionRecord
import cv2
import os

def index(request):
    return render(request, 'index.html')

def detect_debris(request):
    if request.method == "POST":
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        input_path = fs.path(filename)
        output_mask = "static/results/" + filename.replace(".jpg", "_mask.png")

        # 推理
        mask = predict_image(input_path)
        cv2.imwrite(output_mask, mask)

        # 保存记录
        record = DetectionRecord.objects.create(
            image=image,
            result_mask=output_mask
        )

        return JsonResponse({
            'result_url': fs.url(output_mask),
            'area': int(mask.sum() / 255)
        })
```

---

## ✅ 总结：应用场景

| 应用领域 | 说明 |
|---------|------|
| **环保执法** | 自动识别非法倾倒建筑垃圾行为，辅助城管取证 |
| **变化检测** | 对比不同时期航拍图，发现新增垃圾堆 |
| **智慧城市** | 集成于城市大脑，实现低空智能巡查 |
| **灾后评估** | 快速定位废墟区域，支持救援决策 |

---

以上文字及代码仅供参考学习。
