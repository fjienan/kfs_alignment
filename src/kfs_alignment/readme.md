# KFS Alignment (kfs_alignment) ROS 2 功能包开发文档

本项目用于机器人赛场上对 **KFS 立方体** 进行精准视觉锁敌、三维位姿解算以及底盘全向移动自动对齐。功能包集成了 YOLO OBB/Pose 目标检测、基于 OpenCV 的 PnP 算法求解，以及基于分立 PID 的底盘看门狗闭环控制。

---

## 📂 项目目录拓扑结构

```text
kfs_alignment/
├── package.xml                       # ROS 2 包元数据定义
├── setup.py                          # 编译与资源打包配置（自动动态打包 config/launch/models）
├── kfs_alignment/                    # Python 源代码目录
│   ├── __init__.py
│   ├── cube_pose_node.py             # 【视觉节点】负责图像采集、YOLO推理与PnP解算
│   ├── kfs_align_controller_node.py  # 【控制节点】负责姿态欧拉角变换、PID解算及速度控制
│   ├── pnp_solver.py                 # 【数学工具】三维空间点云及正对面三维中心点解算
│   ├── yolo_detector.py              # 【算法推理】YOLO 级联检测器封装
│   └── geometry_utils.py             # 【几何工具】图像及矩阵基础变换函数
├── config/
│   ├── cube_pose_estimator.yaml      # 【核心参数】视觉参数与控制器PID参数一体化集中配置文件
│   └── cube_pose_estimator.rviz      # 【可视化】RViz 3D 环境、网格、Marker 及箭头显示配置
├── launch/
│   └── kfs_align.launch.py           # 【一键启动】同时拉起视觉、控制器与 RViz 的全套脚本
└── models/                           # 深度学习权重存放目录
    ├── obb/
    │   └── yolov8_obb.pt             # YOLO 旋转框权重文件
    └── pose/
        └── yolov8_pose.pt            # YOLO 关键点/姿态权重文件
```

---

## 🚀 快速启动与赛场部署连招

在赛场或实验室上电后，打开全新的终端，严格执行以下命令：

```bash
# 1. 进入真实工作空间目录
cd ~/ares_code_projects

# 2. 编译并拷贝资源文件（带 --cmake-clean-cache 可强制刷新修改后的 YAML 参数）
colcon build --packages-select kfs_alignment --cmake-clean-cache

# 3. 刷新终端当前的 ROS 2 环境变量
source install/setup.bash

# 4. 【一键起飞】拉起视觉、控制器与 RViz 界面
ros2 launch kfs_alignment kfs_align.launch.py
```

---

## 🗺️ 节点与话题通信架构 (Topology)

系统内部通过标准的 ROS 2 话题进行闭环数据传输，核心拓扑设计如下：

### 1. 视觉解算节点 (`cube_pose_node`)

* **输入模式 A (`input.mode: usb`)**：直接绕过 ROS 话题，通过底层 OpenCV 线程强行调用 `/dev/video1` 硬件驱动出图。
* **输入模式 B (`input.mode: topic`)**：订阅其他相机节点发布的图像：
  * **订阅话题**：`/camera/color/image_raw` (`sensor_msgs/msg/Image`)

* **核心输出话题与广播**：

| 输出话题 / 广播名称 | 消息类型 (Message Type) | 作用说明 |
| :--- | :--- | :--- |
| `/cube_pose/pose` | `geometry_msgs/msg/PoseStamped` | 输送给控制器的 KFS 三维绝对坐标与四元数（FLU 坐标系） |
| `/cube_pose/face_corners_3d` | `geometry_msgs/msg/PolygonStamped` | KFS 正对面的 4 个 3D 顶点的物理坐标，用于高级融合 |
| `/cube_pose/visualization` | `sensor_msgs/msg/Image` | 带有渲染圆点、坐标轴、FPS 信息的图像流，输出至 RViz |
| `/cube_pose/markers` | `visualization_msgs/msg/MarkerArray` | 在 RViz 里渲染出真实的虚拟 3D 方块和中心悬浮文本 |
| `TF Broadcaster` | `tf2_ros.TransformBroadcaster` | 持续广播相机基准系到目标系的 TF 树：`rgb_camera_link` $\rightarrow$ `cube_center` |

### 2. 底盘对齐控制节点 (`kfs_align_controller_node`)

* **数据输入**：订阅视觉节点发出的绝对位姿：
  * **订阅话题**：`/cube_pose/pose` (`geometry_msgs/msg/PoseStamped`)

* **核心输出话题**：

| 输出话题名称 | 消息类型 (Message Type) | 作用说明 |
| :--- | :--- | :--- |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | 向底盘下发全向移动速度（含 $v_x, v_y, v_z$ 与角速度 $\omega_z$） |

---

## ⚙️ 核心参数修改终极指南

所有的参数均集中存储于 `config/cube_pose_estimator.yaml` 中。**调车、改分辨率、调 PID 严禁修改源码，只能在此文件内修改！**

### 1. 硬件相机与分辨率参数组 (`usb`)
* `camera_id`: 工业相机在系统中的挂载序号（当前确定为 `1`，对应 `/dev/video1`）。如果提示超时无法出图，可在终端输入 `sudo chmod 777 /dev/video1` 提权。
* `width` & `height`: 相机分辨率。若使用截取窄条技术提升推理帧率，请根据工业相机规格对齐（如 `1280` x `1024`）。
* `use_grayscale`: `True`/`False`。开启后会将图像在传入 YOLO 前转为灰度图，用以过滤环境杂色干扰。

### 2. 目标定义与阈值参数组
* `target.cube_size_mm`: KFS 物理边长的真实毫米数（默认 `100.0`），用于 PnP 相似三角形解算。
* `target.distance`: 机器人最终希望停留在 KFS 正前方多远的距离（单位：米，默认 `1.2`）。
* `target.yaw`: 期望对齐的最终偏航角（默认 `0.0`，视物理安装夹角而定）。

### 3. 底盘限幅与 PID 参数组 (`kfs_align_controller`)
当实车出现震荡、冲刺过猛、刹不住车或反应迟钝时，调整此项：
* `limit.linear_x / y / z`: 限制底盘平移及升降机构的最大输出速度（单位：m/s，推荐赛场安全限制在 `0.3` ~ `0.5` 以内）。
* `limit.angular_z`: 限制最大车头旋转角速度（单位：rad/s）。
* `pid.x`: 前后 PID 参数，格式为 `[Kp, Ki, Kd]`。
* `pid.y`: 左右平移 PID 参数，用于修正横向漂移。
* `pid.yaw`: 车头旋转对齐 PID 参数。如果车子大范围原地左右频繁猛烈摇头，调小第一个 `Kp` 值。

---

## 🛠️ 后续接手开发人员必看规范（关键数学闭环）

开发后续功能或重构代码时，必须遵守以下在真车测试中摸索出来的核心闭环规范，否则会导致系统崩溃或疯狂震荡：

### ⚠️ 规范一：坚决预防“万向节死锁” (Gimbal Lock)
当相机处于倾斜大角度向下注视地面 KFS 时，传统的 `zyx` 顺规欧拉角提取会导致 `Pitch` 轴无限逼近 $-90^\circ$（$-1.51 \text{ rad}$），触发数学死锁，让 `Yaw` 角度计算彻底失真乱跳。
* **开发强制要求**：在 `kfs_align_controller_node.py` 中提取欧拉角时，**必须统一使用 `'xyz'` 顺规**：
  ```python
  # 顺序固定为 xyz，分别对应 roll, pitch, yaw。完美绕开 -90 度死锁点。
  roll, pitch, yaw = r.as_euler('xyz', degrees=False)
  ```

### ⚠️ 规范二：偏航坐标系直角对齐
由于 PnP 视觉求解的特征前向轴与底盘控制器所需的正前向轴（FLU $X$-轴）在物理上存在 $90^\circ$ 的安装夹角，小车物理摆正时得到的裸 `yaw` 物理基准处于 $\pm1.57 \text{ rad}$ 附近。
* **开发强制要求**：计算姿态误差时，由于已在代码中移除了强行硬编码补偿，后续开发者需保证 `Target_Yaw` 与物理基准收敛一致。若更换新相机或底盘导致正对目标时数据整体偏移，请重新采集对齐时的裸数据，并修改 YAML 中的 `target.yaw` 达到闭环。

### ⚠️ 规范三：安全看门狗（Watchdog）与大 dt 限幅
* 控制器内置了 `watchdog_timer`：若超过 `0.5` 秒未收到 `/cube_pose/pose` 话题（如 YOLO 丢帧或相机断开），控制器会强行向 `/cmd_vel` 发布全零速度实行**急停保护**，防止车子失控乱跑。
* 为防止网络卡顿或 YOLO 突发耗时导致单帧 `dt` 异常爆大、憋爆 PID 积分项，代码内强制对时间差进行了安全限幅：
  ```python
  # 强制限制最大周期，防止积分饱和风暴
  dt = np.clip(dt, 0.0, 0.1) 
  ``` |