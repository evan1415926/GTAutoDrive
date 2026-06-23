# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述
GTAutoDrive — AI视觉自动驾驶辅助工具，用于GTA Online。通过OBS虚拟摄像头捕获画面 + 小地图GPS导航 + YOLO目标检测，模拟按键实现自动驾驶和自动避障。

## 环境安装（避免C盘）
- Python 3.10 在 `C:\Users\MI\AppData\Local\Programs\Python\Python310`（系统已装，不动）
- OBS Studio 在 `D:\obs-studio`（系统已装，不动）
- **虚拟环境必须建在G盘**: `python -m venv G:\GTAutoDrive\venv`
- 激活: `G:\GTAutoDrive\venv\Scripts\activate`
- pip 安装依赖: `pip install -r requirements.txt`
- 本项目的所有 pip 包都装在 venv 中，不影响 C 盘
- **注意**: Windows 系统代理设置为 `127.0.0.1:7897`（Clash），必须先启动代理才能 pip install
- 如果 venv 的 pip 有 SSL 问题，用系统 Python 直装: 
  ```
  export HTTP_PROXY="http://127.0.0.1:7897"
  export HTTPS_PROXY="http://127.0.0.1:7897"
  C:/.../Python310/python -m pip install --target G:/GTAutoDrive/venv/Lib/site-packages -r requirements.txt
  ```

## OBS 屏幕捕获方案
- OBS 通过 Game Capture 或 Display Capture 捕获 GTA V 画面
- 启用 OBS 虚拟摄像头（Tools → VirtualCam → Start）
- Python 通过 `cv2.VideoCapture(0)` 读取虚拟摄像头画面（注意确认设备索引）
- `capture/screen_capture.py` 负责从摄像头读取帧，再按配置裁剪出主画面区域和小地图区域
- OBS 配置要点：
  - 输出分辨率设为显示器原生分辨率（如 1920x1080）
  - 确保 GTA V 以无边框窗口模式运行
  - 虚拟摄像头帧率设为 30 FPS

## 构建/运行命令
```bash
# 激活虚拟环境
G:\GTAutoDrive\venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 运行主程序
python main.py

# 测试工具
python tools/test_capture.py       # 测试OBS虚拟摄像头读取
python tools/test_detection.py     # 静态截图测试YOLO
python tools/test_minimap.py       # 截图测试小地图处理
python tools/test_input.py         # 记事本中测试按键模拟
python tools/calibrate_regions.py  # 交互式标定捕获区域

# 单元测试
pytest tests/
```

## 核心架构

### 数据流（每帧）
```
OBS VirtualCam → cv2.VideoCapture.read() → full_frame BGR
  → crop_main(full_frame) → main_frame → ObjectDetector.detect() → List[Detection]
  → crop_minimap(full_frame) → minimap_frame → MinimapProcessor.process() → MinimapData
  → ControlDecisionMaker.decide(MinimapData, Detections, w, h) → ControlActions
  → InputSimulator.apply(ControlActions) → 按键事件
```

### 模块职责
1. **capture/screen_capture.py** — 通过 OpenCV `cv2.VideoCapture` 读取 OBS 虚拟摄像头，按配置裁剪出主画面(驾驶视野)和小地图区域
2. **vision/object_detector.py** — YOLOv8n 检测驱动相关对象（car/motorcycle/bus/truck/person/bicycle/traffic light/stop sign），过滤后按置信度排序
3. **vision/minimap_processor.py** — HSV `(20,100,120)-(40,255,255)` 提取黄色GPS路线。**关键假设**: 标准GTA V小地图玩家居中、箭头始终指向上方（向上=前进方向）。路径搜索从玩家位置向上在GPS mask中找第一个亮像素
4. **control/decision_maker.py** — 三层优先级融合:
   - 紧急: 障碍物过近 → 全力刹车
   - 避障: 障碍物在路径中 → 绕开+减速
   - 导航: 跟随GPS路线 → PID风格转向
   - 输出经EMA平滑(alpha可配)，避免方向突变
5. **input/input_simulator.py** — 维护内部按键状态字典，仅状态变化时发送press/release，避免每帧重复触发
6. **config/settings.py** — AppConfig 数据类，所有可调参数集中管理，支持JSON序列化

### 关键可调参数 (config/default_config.json)
- `capture.camera_index` — OBS虚拟摄像头设备索引（默认0，需用 test_capture.py 确认）
- `control.steer_sensitivity` — 转向灵敏度 (0.6)
- `control.obstacle_brake_distance_norm` — 障碍物刹车距离阈值 (0.18 屏幕比例)
- `control.emergency_brake_distance_norm` — 紧急刹车距离 (0.08)
- `control.steer_smoothing_alpha` — 转向平滑系数 (0.3)
- `minimap.hsv_yellow_lower/upper` — GPS路线黄色HSV阈值
- `detection.confidence_threshold` — YOLO置信度阈值 (0.35)
- `capture.main_region/minimap_region` — 画面裁剪区域（相对OBS输出画面的比例，0-1）

### 项目结构
```
G:\GTAutoDrive\
├── main.py                    # 入口，主循环GTAutoDrive类
├── requirements.txt
├── venv/                      # Python虚拟环境（G盘）
├── config/
│   ├── settings.py            # AppConfig 数据类
│   └── default_config.json
├── capture/
│   └── screen_capture.py      # OBS虚拟摄像头读取+裁剪
├── vision/
│   ├── object_detector.py     # YOLO检测封装
│   └── minimap_processor.py   # GPS路线提取
├── control/
│   └── decision_maker.py      # 驾驶决策融合
├── input/
│   └── input_simulator.py     # 按键模拟（状态追踪）
├── models/
│   └── data_models.py         # Detection, MinimapData, ControlActions
├── utils/
│   ├── debug_visualizer.py
│   ├── fps_counter.py
│   └── window_utils.py
├── tools/
│   ├── calibrate_regions.py
│   ├── test_capture.py
│   ├── test_detection.py
│   ├── test_minimap.py
│   └── test_input.py
└── tests/
    ├── test_decision_maker.py
    ├── test_minimap_processor.py
    └── fixtures/
```

## 安全机制（必须遵守）
- 全局紧急停止热键 Ctrl+Shift+Q，立即释放所有按键
- 任何退出路径（正常/异常）都必须调用 `cleanup()` 释放按键
- 油门上限 0.8，永不全油门
- 修改 `input/input_simulator.py` 时必须验证 release_all() 在 finally 块中有调用
