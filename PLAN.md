# GTAutoDrive - AI视觉自动驾驶辅助工具

## Context
帮助用户在GTA Online中实现自动驾驶——跟随小地图GPS导航路线行驶，自动避让车辆和行人障碍物。
目录 `G:\GTAutoDrive` 目前为空，从零开始搭建。

## 环境约束
- **C盘仅剩 11G**，Python依赖全部装到项目 venv（G盘，191G可用）
- **OBS Studio** 已安装在 `D:\obs-studio`
- **Python 3.10** 已装在C盘（不动）
- 虚拟环境: `python -m venv G:\GTAutoDrive\venv`

## 技术选型
- **语言**: Python 3.10
- **屏幕捕获**: OBS Virtual Camera（Game Capture捕获GTA V）→ OpenCV `cv2.VideoCapture` 读取
- **目标检测**: YOLOv8n (ultralytics, COCO预训练权重)
- **小地图处理**: OpenCV HSV颜色阈值提取黄色GPS路线
- **按键模拟**: pynput（按键状态追踪）
- **紧急停止**: keyboard 全局热键 Ctrl+Shift+Q

## OBS 捕获方案
OBS 已安装在 `D:\obs-studio`，使用虚拟摄像头功能：
1. OBS 添加 Game Capture 或 Display Capture 源捕获 GTA V 画面
2. 启用 Virtual Camera (Tools → VirtualCam → Start)
3. Python 通过 `cv2.VideoCapture(camera_index)` 读取虚拟摄像头帧
4. 在捕获的完整画面中裁剪出主画面（驾驶视野）和小地图区域
5. 必需工具: OBS 本地安装自带 VirtualCam（无需额外插件）
6. GTA V 建议设置为无边框窗口模式以确保捕获兼容性

## 核心架构

### 数据流（每帧）
```
OBS VirtualCam → cv2.VideoCapture.read() → full_frame BGR
  → crop_main(full_frame) → main_frame → YOLO检测 → List[Detection]
  → crop_minimap(full_frame) → minimap_frame → HSV提取GPS路线 → MinimapData
  → ControlDecisionMaker.decide(MinimapData, Detections) → ControlActions
  → InputSimulator.apply(ControlActions) → WASD按键事件
```

### 模块设计

**capture/screen_capture.py** — OBS虚拟摄像头读取封装
- `ScreenCapture.__init__(camera_index, main_region, minimap_region)` — main_region/minimap_region 是相对完整画面的裁剪比例(0-1)
- `capture_both()` → 读完一帧，按比例裁剪返回 (main_frame, minimap_frame)
- 启动时检测摄像头是否可用，打印设备名确认

**vision/object_detector.py** — YOLOv8n 目标检测封装
- COCO类别过滤到驾驶相关: car/motorcycle/bus/truck/person/bicycle/traffic light/stop sign
- 输入 BGR frame，输出 List[Detection]（含bbox、置信度、类别）
- 辅助方法: `filter_in_driving_path()` 按横向位置过滤路径内障碍物

**vision/minimap_processor.py** — 小地图GPS路线提取
- **关键假设**: GTA V标准小地图玩家箭头居中且始终指向上方（向上=前进方向）
- HSV色域 `(20,100,120)-(40,255,255)` 提取黄色GPS路线mask
- 从玩家位置向上在mask中搜索第一个亮像素作为下一个路径点
- 搜索锥角 ±60°，逐渐扩大
- 输出 MinimapData: 玩家位置、路径方向角、横向偏移量、目标距离

**control/decision_maker.py** — 驾驶决策融合
- 三层优先级:
  1. 紧急制动: 障碍物距离 < 0.08*屏幕高度 → 全力刹车
  2. 避障绕行: 障碍物在行驶路径中 → 减速+绕开
  3. GPS导航: 跟随路径方向 → PID风格转向
- 所有输出经EMA平滑: `steer = α*new + (1-α)*prev`
- steering 计算: `clamp(path_direction_error * sensitivity / 30, -1, 1)`
- 避障转向: `steer_avoid = (0.5 - obstacle_center_x_norm) * 2.0`

**input/input_simulator.py** — 按键状态追踪
- 维护 `{throttle, brake, left, right, handbrake}` 按键状态字典
- 仅状态变化时调用 `pynput.keyboard.Controller.press()/release()`
- `release_all()` 和 `emergency_stop()` 用于安全退出

**config/settings.py** — AppConfig 数据类
- 所有模块配置集中在一个嵌套数据类中
- 支持 `from_json()` / `to_json()` 持久化
- 可配置项: 摄像头索引、裁剪区域、HSV阈值、检测阈值、控制参数、按键绑定

## 项目结构
```
G:\GTAutoDrive\
├── main.py                     # 入口，GTAutoDrive类，主循环
├── requirements.txt
├── venv/                       # Python虚拟环境
├── config/
│   ├── __init__.py
│   ├── settings.py             # AppConfig 数据类 + JSON序列化
│   └── default_config.json
├── capture/
│   ├── __init__.py
│   └── screen_capture.py       # OBS虚拟摄像头读取+裁剪
├── vision/
│   ├── __init__.py
│   ├── object_detector.py      # YOLOv8n 检测封装
│   └── minimap_processor.py    # GPS路线HSV提取
├── control/
│   ├── __init__.py
│   └── decision_maker.py       # 导航+避障融合决策
├── input/
│   ├── __init__.py
│   └── input_simulator.py      # pynput按键模拟（状态追踪）
├── models/
│   ├── __init__.py
│   └── data_models.py          # Detection, MinimapData, ControlActions
├── utils/
│   ├── __init__.py
│   ├── debug_visualizer.py     # cv2.imshow 三窗口调试
│   ├── fps_counter.py          # 滚动平均FPS
│   └── window_utils.py         # 摄像头枚举、区域检测
├── tools/
│   ├── calibrate_regions.py    # 交互式标定裁剪区域
│   ├── test_capture.py         # 测试OBS虚拟摄像头读取
│   ├── test_detection.py       # 静态截图测试YOLO
│   ├── test_minimap.py         # 截图测试小地图GPS提取
│   └── test_input.py           # 记事本测试按键模拟
└── tests/
    ├── __init__.py
    ├── test_decision_maker.py
    ├── test_minimap_processor.py
    └── fixtures/
```

## 实施顺序（4个阶段）

### Phase 1: 基础设施
1. 创建虚拟环境: `python -m venv G:\GTAutoDrive\venv`
2. 创建项目目录结构
3. 编写 `requirements.txt`
4. 实现 `config/settings.py` 和 `models/data_models.py`
5. 实现 `capture/screen_capture.py` (OBS虚拟摄像头)
6. 实现 `input/input_simulator.py` (pynput)
7. 编写 `test_capture.py` 和 `test_input.py` 验证

### Phase 2: 视觉模块
8. 实现 `vision/minimap_processor.py` (GPS路线提取)
9. 实现 `vision/object_detector.py` (YOLOv8n)
10. 编写 `test_minimap.py` 和 `test_detection.py` 调试

### Phase 3: 控制与集成
11. 实现 `control/decision_maker.py` (决策融合)
12. 实现 `main.py` (主循环、紧急停止、调试可视化)
13. 单元测试 decision_maker 和 minimap_processor

### Phase 4: 调优
14. `tools/calibrate_regions.py` 交互式区域标定
15. 在GTA V中实车测试，调优控制参数和HSV阈值

## 验证方式
- **单模块测试**: 各 `test_*.py` 独立验证
- **单元测试**: `pytest tests/` 验证决策逻辑
- **干运行模式**: 调试模式打印控制决策，不发送按键
- **实车测试**: 空地→高速→弯道→城市→车流，逐级增加难度

## 依赖项 (requirements.txt)
```
opencv-python==4.10.0.84
numpy==1.26.4
ultralytics==8.2.0
torch==2.3.0
torchvision==0.18.0
pynput==1.7.7
keyboard==0.13.5
Pillow==10.3.0
screeninfo==0.8.1
matplotlib==3.9.0
```
注意: 去掉了 mss（用OBS替代），屏幕捕获只依赖 OpenCV 的 `cv2.VideoCapture`。
