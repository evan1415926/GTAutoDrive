# GTAutoDrive

AI 视觉自动驾驶辅助工具，适用于 GTA V / GTA Online。

**理念**：模仿现实自动驾驶范式 — 先感知场景（障碍物、车道线），再基于量化规则做驾驶决策。

## 架构

```
1920x1080 全屏截图 (OBS VirtualCam)
    │
    ├──→ YOLOv8n 目标检测 → 障碍物列表
    │     (car/truck/bus/person/motorcycle/bicycle + 距离估算)
    │
    └──→ HSV 车道提取 → 鸟瞰透视 → 多项式拟合 → 车道信息
          (左右车道线曲率 + 偏离中心距离)

                         ↓

              规则规划器 (Planner)

    1. 紧急刹车  — 前方障碍物过近           → S
    2. 避障减速  — 前方有车在车道内         → NONE (松油滑行)
    3. 车道居中  — 偏离车道中心超过阈值     → W+A/D
    4. 弯道跟随  — 检测到弯道             → W+A/D
    5. 默认巡航  — 以上都不满足            → W

                         ↓
                  WASD 按键输出
```

## 环境要求

- **Windows 10/11**
- **Python 3.10+**
- **OBS Studio** + VirtualCam 插件（捕获 GTA V 画面）
- **GTA V** 以无边框窗口模式运行
- NVIDIA GPU（可选，YOLO 会自动用 CUDA）

## 安装

```bash
# 1. 克隆仓库
git clone git@github.com:evan1415926/GTAutoDrive.git
cd GTAutoDrive

# 2. 创建虚拟环境
python -m venv venv
venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

首次运行会自动下载 YOLOv8n 模型（约 6MB）。

## 使用方法

### 1. 启动 OBS VirtualCam

- OBS → Tools → VirtualCam → Start
- 确保输出分辨率设为 1920x1080，30 FPS
- GTA V 以无边框窗口运行

### 2. 测试感知模块（推荐先跑）

```bash
python tools/test_detection.py     # 验证 YOLO 目标检测是否正常
python tools/test_lane.py          # 验证车道线检测是否正常
```

### 3. 标定车道线（重要）

车道线不准时使用交互式标定工具：

```bash
python tools/calibrate_lane.py
```

拖动 trackbar 直到红蓝线条贴合实际车道线，按 **S** 保存配置。

### 4. 运行主程序

```bash
python main.py                    # 感知+规划 模式（默认）
python main.py --dry-run          # 仅显示画面，不模拟按键
python main.py --model xxx.pt     # BC 模型回退模式（旧版）
```

**热键：**
- `F8` — 紧急停止（释放所有按键并退出）
- `Q` — 在调试窗口中也可退出

## 项目结构

```
GTAutoDrive/
├── main.py                      # 主入口，运行主循环
├── train.py                     # BC 模型训练入口（旧版）
├── requirements.txt
│
├── config/
│   └── settings.py              # 所有可调参数（数据类）
│
├── capture/
│   └── screen_capture.py        # OBS VirtualCam 读取
│
├── vision/
│   ├── object_detector.py       # YOLOv8n 目标检测
│   └── lane_detector.py         # HSV 车道线提取 + 透视变换 + 多项式拟合
│
├── control/
│   └── planner.py               # 量化规则决策（5 级优先级）
│
├── input/
│   └── input_simulator.py       # 按键状态追踪 + 模拟
│
├── models/
│   └── data_models.py           # Detection / LaneInfo / ControlAction
│
├── model/                        # 旧版 BC 行为克隆（不再主动开发）
│   ├── network.py
│   ├── dataset.py
│   └── trainer.py
│
├── data/                         # 数据加载 + 类别平衡
│   ├── loader.py
│   └── balancer.py
│
├── tools/
│   ├── calibrate_lane.py         # 🔧 车道线透视标定（实时 trackbar）
│   ├── test_detection.py         # YOLO 检测测试
│   ├── test_lane.py              # 车道检测测试
│   └── test_capture.py           # OBS 捕获测试
│
├── tests/                        # 单元测试（40个）
│   ├── test_config.py
│   ├── test_input_simulator.py
│   ├── test_balancer.py
│   ├── test_loader.py
│   ├── test_dataset.py
│   ├── test_network.py
│   ├── test_trainer.py
│   ├── test_collector.py
│   └── test_integration.py
│
└── utils/
    ├── fps_counter.py
    └── debug_overlay.py
```

## 关键可调参数

所有参数集中在 `config/settings.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PerceptionConfig.detection_conf` | 0.35 | YOLO 置信度阈值 |
| `PerceptionConfig.lane_roi_top` | 0.55 | 车道 ROI 起始位置（画面顶部裁掉比例） |
| `PerceptionConfig.lane_persp_top_y` | 0.40 | 透视消失点 Y 坐标 |
| `PlannerConfig.emergency_brake_dist` | 0.10 | 紧急刹车距离阈值 |
| `PlannerConfig.obstacle_brake_dist` | 0.22 | 避障减速距离阈值 |
| `PlannerConfig.lane_offset_thresh` | 0.25 | 车道居中修正阈值 |
| `PlannerConfig.steer_gain` | 2.5 | 转向修正强度 |
| `PlannerConfig.curve_steer_gain` | 0.8 | 弯道跟随强度 |

## 已知问题 & 待改进

- [ ] **车道线透视参数需标定**：默认参数不一定适用于所有 GTA V 视角，需用 `calibrate_lane.py` 标定
- [ ] **HSV 车道检测对光照敏感**：夜间、雨天、隧道内可能检测失败
- [ ] **GTA V 小地图 GPS 导航未接入**：目前只做 L2 车道保持 + 避障，未接入导航路线
- [ ] **交叉路口无车道线时无策略**：目前降级为直行
- [ ] **速度控制只有 W/NONE/S 三档**：没有精细油门控制
- [ ] **BC 模型准确率卡在 75%**：端到端行为克隆方案已放弃，仅保留为 `--model` 可选回退

## 贡献

欢迎 PR！建议从上面的待改进列表入手。改动前建议先跑现有测试：

```bash
pytest tests/
```

## License

MIT
