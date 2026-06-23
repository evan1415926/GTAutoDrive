# GTA V L2 自动驾驶 — 端到端行为克隆 设计文档

日期: 2026-06-23

## 1. 目标与范围

### 核心目标
训练一个端到端 CNN，输入 GTA V 游戏画面，输出驾驶动作，实现 L2 级不变道巡航。

### v1 范围（当前）
- 在当前车道内稳定巡航，不压线
- 自动跟车：前车减速 → 本车减速，前车加速 → 恢复巡航速度
- 弯道自动减速、直线恢复巡航速度
- 线下模式（单机），多车型混合训练

### v1 不做
- 变道、超车、路口转弯、按导航行驶
- 识别红绿灯、交通标志牌
- 小路 / 越野 / 逆行道路
- 线上模式（GTA Online）

### 未来演进方向
CNN 负责低层驾驶直觉（10-30Hz），VLM/多模态大模型周期性地输出高层指令（变道、超车、导航），两层架构。

## 2. 技术路线

端到端行为克隆（模仿 Sentdex 2017 路线，2026 工具链重做）。

### 输入/输出
- 输入: GTA V 游戏画面，320×180×3 RGB
- 输出: 7 分类 — W, WA, WD, A, D, S, NONE
- argmax 后直接映射为对应按键

### 为什么选分类而非回归
键盘控制是离散的（每按键要么按要么不按）。分类输出直接匹配这个物理现实，避免连续值→离散化的转换损耗。GTA V 物理引擎自身阻尼足以让分类切换产生流畅驾驶。

### 为什么选端到端而非多任务
- 被 Sentdex / NVIDIA PilotNet / comma.ai openpilot 反复验证
- 多任务学习的 loss 平衡、标注噪声、负迁移风险在单人项目中得不偿失
- Grad-CAM 热力图可实现可解释性，黑盒不是真痛点

## 3. 架构

```
                    训练阶段
你开车(WASD) → 录制(画面, 按键) → 平衡数据 → 训练CNN

                    推理阶段
GTA V 画面 → dxcam 截屏 → 缩放 320×180 → CNN → 7类 softmax
  → EMA 平滑 → argmax → 对应按键 → GTA V
```

## 4. 模型

基于 torchvision ResNet-18 改造:

```
输入: 320×180×3 RGB
  │
  ▼
Conv 7×7, stride=2, 64ch + BN + ReLU
MaxPool 3×3, stride=2
  │
ResBlock 64ch  ×2
ResBlock 128ch ×2  (stride=2)
ResBlock 256ch ×2  (stride=2)
  │
Global Average Pooling  →  256-d
  │
Dropout 0.5
FC 256 → 7  →  Softmax
```

- 参数量约 11M（标准 ResNet-18），GTX 1060 能跑 30fps+
- 不加载 ImageNet 预训练权重（GTA 画面分布与自然图像差异大）
- 输入 320×180 对道路纹理/车辆/车道线完全可分辨

## 5. 数据采集

### 录制规格
| 参数 | 值 |
|------|-----|
| 采集帧率 | 10 FPS |
| 画面尺寸 | 320×180 |
| 标签格式 | 7 分类 |
| 数据量目标 | 正常驾驶 10万帧 + 纠错 3万帧 |

### 操作方式
- `F5`: 开始/暂停录制（屏幕红色圆点提示）
- `F6`: 切换训练模式 / 纠错模式

### 场景覆盖
| 场景 | 说明 |
|------|------|
| 高速公路直线 | 主要数据，80-130 km/h 巡航 |
| 高速弯道 | 进弯收油 → 弯中稳方向 → 出弯加油 |
| 城市直路 | 中低速，遇到路口减速 |
| 跟车行驶 | 前方有 NPC 车辆，加减速保持车距 |
| 夜晚/雨天 | 约 10%，覆盖光照变化 |
| 纠错-偏左回正 | 偏离左侧 → 向右修正 |
| 纠错-偏右回正 | 偏离右侧 → 向左修正 |
| 纠错-逼近减速 | 快速接近前车 → 松油 + 刹车 |

### 多车型方案
用 3-5 辆不同类型车（跑车、轿车、SUV）各录 5-10 分钟，混合训练一个通用模型。

## 6. 数据平衡

训练前按 7 类别统计帧数，以最少类为基准，其他类随机下采样至相同数量。防止"永远输出 W"。

## 7. 训练

### 数据划分
80% 训练 / 20% 验证

### 数据增强（训练时随机应用）
| 操作 | 参数 | 目的 |
|------|------|------|
| 亮度/对比度抖动 | 80%-120% | 光照、天气 |
| 左右平移 | -20px ~ +20px | 模拟偏航 |
| 水平翻转 + 标签镜像 | 50% | WA↔WD, A↔D |
| 小角度旋转 | -3° ~ +3° | 路面倾斜 |

### 训练配置
| 参数 | 值 |
|------|-----|
| Loss | Cross-Entropy（7 分类） |
| Optimizer | Adam, lr=1e-3 |
| 学习率策略 | ReduceLROnPlateau, patience=5, min=1e-5 |
| Batch Size | 64 |
| Epochs | 最多 100, EarlyStopping patience=15 |
| 设备 | CUDA 优先 |

### 评估
- 安全类召回率: S/A/D 三类至少 90%
- 时序平滑度: 连续 50 帧内类别切换次数

## 8. 推理

### 每帧流程
```
dxcam 截取 GTA 窗口 → resize 320×180 → CNN 推理 → EMA(α=0.4) → argmax → 按键
```

### EMA 防抖
probs_smoothed = 0.4 × probs_new + 0.6 × probs_prev

### 安全机制
- `F8` 紧急停止，立即释放所有按键
- 异常退出 finally 块调用 release_all()
- 油门上限 0.8（永不全油门）

## 9. 项目目录结构

```
G:\GTAutoDrive\
├── main.py                     # 推理主循环入口
├── train.py                    # 训练入口
├── collect.py                  # 数据采集入口
├── requirements.txt
├── config/
│   └── settings.py             # 所有配置集中管理
├── capture/
│   └── screen_capture.py       # dxcam 截取 GTA 窗口
├── input/
│   └── input_simulator.py      # pynput 按键状态追踪+模拟
├── model/
│   ├── network.py              # CNN 模型定义 (改造 ResNet-18)
│   ├── dataset.py              # DataLoader + 数据增强
│   └── trainer.py              # 训练循环 + 验证
├── data/
│   ├── collector.py            # 录制 (画面, 按键)
│   ├── balancer.py             # 类别平衡预处理
│   └── loader.py               # 从磁盘加载训练数据
├── data/recordings/            # 采集的原始数据
├── data/models/                # 训好的 .pt 文件
├── tools/
│   └── test_capture.py         # 截屏 + 按键快速验证
└── utils/
    ├── fps_counter.py          # FPS 统计
    └── debug_overlay.py        # 画面叠加状态信息
```

### 删除的旧模块
砍掉: `vision/`, `control/`, `training/`, `models/data_models.py`, `gui.py`,
`config/default_config.json`, `tools/calibrate_regions.py`, `tools/test_detection.py`,
`tools/test_minimap.py`, `debug_dumps/`, `test_printwindow.png`, `yolov8n.pt`,
`__pycache__/`

## 10. 复用第三方库

| 功能 | 库 |
|------|-----|
| 屏幕捕获 | `dxcam` (DXGI desktop duplication) |
| 键盘模拟 | `pynput` (sendinput 级别) |
| 热键监听 | `keyboard` |
| 模型 backbone | `torchvision.models.resnet18` |
| 图像增强 | `torchvision.transforms` / `albumentations` |
| 数组处理 | `numpy` |
| 图像读写 | `opencv-python` / `Pillow` |

## 11. 实施顺序

### Phase 1: 基础设施
1. 清理旧代码，建立新目录结构
2. `config/settings.py` 配置数据类
3. `capture/screen_capture.py` dxcam 封装
4. `input/input_simulator.py` pynput 按键模拟（状态追踪）
5. `tools/test_capture.py` 验证捕获+按键

### Phase 2: 数据采集
6. `data/collector.py` 录制系统（F5 控制，模式切换）
7. `collect.py` 入口
8. 实车录制数据

### Phase 3: 模型与训练
9. `model/network.py` CNN 模型定义
10. `model/dataset.py` DataLoader + 数据增强
11. `data/balancer.py` 类别平衡
12. `model/trainer.py` 训练循环
13. `train.py` 入口

### Phase 4: 推理与调优
14. `main.py` 推理主循环
15. `utils/` 调试工具
16. 线下实车测试，调优参数
