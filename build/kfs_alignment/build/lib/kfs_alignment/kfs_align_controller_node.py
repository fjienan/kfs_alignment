#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation as R
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, error, dt):
        if dt <= 0.0:
            return 0.0
        self.integral += error * dt
        # 积分限幅抗饱和
        self.integral = np.clip(self.integral, -self.output_limit, self.output_limit)
        
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.prev_error = error
        return np.clip(output, -self.output_limit, self.output_limit)

class KFSAlignController(Node):
    def __init__(self):
        super().__init__('kfs_align_controller')

        # === 1. 声明并读取参数 ===
        self.declare_parameter("target.distance", 0.7)     # 目标停止距离 (米)
        self.declare_parameter("target.yaw", 0.0)          # 目标偏航角
        self.declare_parameter("target.y_offset", 0.0)     # Y轴物理偏置，调整左右归零点
        
        # 限制底盘最大速度
        self.declare_parameter("limit.linear_x", 0.8)      # m/s
        self.declare_parameter("limit.linear_y", 0.8)      # m/s
        self.declare_parameter("limit.linear_z", 0.5)      # m/s
        self.declare_parameter("limit.angular_z", 1.0)     # rad/s

        # PID 参数
        self.declare_parameter("pid.x", [1.5, 0.0, 0.1])
        self.declare_parameter("pid.y", [1.5, 0.0, 0.15])
        self.declare_parameter("pid.z", [2.0, 0.0, 0.2])
        self.declare_parameter("pid.yaw", [1.2, 0.0, 0.1])

        self._load_parameters()

        # === 2. 初始化 ROS 接口 ===
        self.cmd_pub = self.create_publisher(Float32MultiArray, 't0x0101_action', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/cube_pose/pose', self.pose_callback, 10)

        self.last_time = self.get_clock().now()
        self.last_pose_time = self.get_clock().now()
        
        # 安全看门狗
        self.watchdog_timer = self.create_timer(0.1, self.watchdog_callback)

        self.get_logger().info("KFS 视觉对齐节点已启动！采用标准 FLU 及 [vx, vy, wz] 数组下发。")

    def _load_parameters(self):
        self.target_dist = self.get_parameter("target.distance").value
        self.target_yaw = self.get_parameter("target.yaw").value
        self.target_y_offset = self.get_parameter("target.y_offset").value
        
        lim_x = self.get_parameter("limit.linear_x").value
        lim_y = self.get_parameter("limit.linear_y").value
        lim_z = self.get_parameter("limit.linear_z").value
        lim_yaw = self.get_parameter("limit.angular_z").value

        pid_x_params = self.get_parameter("pid.x").value
        pid_y_params = self.get_parameter("pid.y").value
        pid_z_params = self.get_parameter("pid.z").value
        pid_yaw_params = self.get_parameter("pid.yaw").value

        self.pid_x = PIDController(*pid_x_params, lim_x)
        self.pid_y = PIDController(*pid_y_params, lim_y)
        self.pid_z = PIDController(*pid_z_params, lim_z)
        self.pid_yaw = PIDController(*pid_yaw_params, lim_yaw)

    def pose_callback(self, msg: PoseStamped):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time
        self.last_pose_time = current_time
        
        dt = np.clip(dt, 0.0, 0.1)
        if dt <= 0:
            return

        # ========================================================
        # 1. 提取上游已经转换好的 FLU 坐标
        # ========================================================
        raw_x = msg.pose.position.x  # 前后深度
        raw_y = msg.pose.position.y  # 左右偏离
        
        # 提取旋转 (上游已经将其转至 FLU 坐标系)
        q = [msg.pose.orientation.x, msg.pose.orientation.y, 
             msg.pose.orientation.z, msg.pose.orientation.w]
        rx, ry, rz = R.from_quat(q).as_euler('xyz', degrees=False)
        raw_yaw = rz  # FLU 中绕 Z 轴即为 Yaw
        
        # ========================================================
        # 2. 物理翻转开关 (治各种不服)
        # ========================================================
        current_x = raw_x         # 深度一般没问题
        current_y = raw_y        # 👈 你观测到左边是负数，所以加个负号强制归正
        current_yaw = raw_yaw + np.pi/2     # 偏航角暂不翻转，如果底盘自转反了，改成 -raw_yaw

        # 计算误差
        error_x = current_x - self.target_dist 
        error_y = current_y - self.target_y_offset 
        error_yaw = (self.target_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi

        # ========================================================
        # 3. PID 计算
        # ========================================================
        out_vx = self.pid_x.compute(error_x, dt)
        out_vy = self.pid_y.compute(error_y, dt)
        out_wz = self.pid_yaw.compute(error_yaw, dt)

        # ========================================================
        # 4. 打包发送 (严格遵照队友代码的底层数组解析顺序)
        # ========================================================
        action_msg = Float32MultiArray()
        
        # 队友底层逻辑：坑位0是vx，坑位1是vy，坑位2是wz
        action_msg.data = [float(out_vx), float(out_vy), float(out_wz)] 
        
        self.cmd_pub.publish(action_msg)

        self.get_logger().info(
            f"Err-> X:{error_x:.2f} Y:{error_y:.2f} Yaw:{error_yaw:.2f} || 发送:[vx:{out_vx:.2f}, vy:{out_vy:.2f}, wz:{out_wz:.2f}]"
        )

    def watchdog_callback(self):
        time_since_last_pose = (self.get_clock().now() - self.last_pose_time).nanoseconds / 1e9
        if time_since_last_pose > 0.5:
            # 丢失目标超过 0.5 秒，下发全零数组紧急停车
            stop_msg = Float32MultiArray()
            stop_msg.data = [0.0, 0.0, 0.0]
            self.cmd_pub.publish(stop_msg)

def main(args=None):
    rclpy.init(args=args)
    node = KFSAlignController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()