"""ROS 2 node: fuse vision pose with IMU using a lightweight Kalman filter.

Design goals:
- Minimal dependencies (numpy + rclpy + standard msgs)
- Default IMU input is a topic (sensor_msgs/Imu)
- Publish fused PoseStamped for downstream use

Notes:
- Position KF runs in the pose frame (assumed consistent across measurements).
- IMU linear acceleration fusion is OFF by default (frame alignment is non-trivial).
- Orientation fusion is a simple quaternion slerp between vision and IMU orientations.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Imu


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return (q / n).astype(np.float64)


def _quat_dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.reshape(4), b.reshape(4)))


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions [x,y,z,w]."""
    t = float(np.clip(t, 0.0, 1.0))
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)

    dot = _quat_dot(q0, q1)
    # Take shortest path
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # If very close, lerp is fine
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return _quat_normalize(q)

    theta_0 = math.acos(float(np.clip(dot, -1.0, 1.0)))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)

    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _quat_normalize((s0 * q0) + (s1 * q1))


@dataclass
class KFState:
    x: np.ndarray  # (6,) [p(3), v(3)]
    P: np.ndarray  # (6,6)
    t_sec: float


class LinearKFPos3D:
    """Constant-velocity Kalman filter with optional acceleration control input."""

    def __init__(self, sigma_accel: float, sigma_meas: float) -> None:
        self._sigma_accel = float(sigma_accel)
        self._sigma_meas = float(sigma_meas)
        self._state: Optional[KFState] = None

    def reset(self) -> None:
        self._state = None

    @property
    def initialized(self) -> bool:
        return self._state is not None

    def initialize(self, pos_m: np.ndarray, t_sec: float) -> None:
        x = np.zeros((6,), dtype=np.float64)
        x[0:3] = pos_m.reshape(3).astype(np.float64)
        P = np.eye(6, dtype=np.float64) * 1e-3
        P[0:3, 0:3] = np.eye(3) * (self._sigma_meas**2)
        P[3:6, 3:6] = np.eye(3) * 1.0
        self._state = KFState(x=x, P=P, t_sec=float(t_sec))

    def predict(self, t_sec: float, accel_mps2: Optional[np.ndarray] = None) -> None:
        if self._state is None:
            return

        dt = float(t_sec - self._state.t_sec)
        if dt <= 0.0:
            return

        A = np.eye(6, dtype=np.float64)
        A[0:3, 3:6] = np.eye(3) * dt

        # Optional control input (acceleration)
        u = np.zeros((3,), dtype=np.float64)
        if accel_mps2 is not None:
            u = accel_mps2.reshape(3).astype(np.float64)
        B = np.zeros((6, 3), dtype=np.float64)
        B[0:3, :] = np.eye(3) * (0.5 * dt * dt)
        B[3:6, :] = np.eye(3) * dt

        # Discrete white-noise acceleration model
        sa2 = self._sigma_accel**2
        q11 = (dt**4) / 4.0 * sa2
        q12 = (dt**3) / 2.0 * sa2
        q22 = (dt**2) * sa2
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0:3, 0:3] = np.eye(3) * q11
        Q[0:3, 3:6] = np.eye(3) * q12
        Q[3:6, 0:3] = np.eye(3) * q12
        Q[3:6, 3:6] = np.eye(3) * q22

        x = A @ self._state.x + B @ u
        P = A @ self._state.P @ A.T + Q
        self._state = KFState(x=x, P=P, t_sec=float(t_sec))

    def update_pos(self, pos_m: np.ndarray, t_sec: float) -> None:
        """Update using a position measurement at time t_sec (predict to t_sec before calling)."""
        if self._state is None:
            self.initialize(pos_m, t_sec)
            return

        # Measurement: z = H x + v, where H maps position
        H = np.zeros((3, 6), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)
        R = np.eye(3, dtype=np.float64) * (self._sigma_meas**2)

        z = pos_m.reshape(3).astype(np.float64)
        y = z - (H @ self._state.x)
        S = H @ self._state.P @ H.T + R
        K = self._state.P @ H.T @ np.linalg.inv(S)

        x = self._state.x + K @ y
        I = np.eye(6, dtype=np.float64)
        P = (I - K @ H) @ self._state.P
        self._state = KFState(x=x, P=P, t_sec=float(t_sec))

    def get_pos_vel(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._state is None:
            return np.zeros((3,), dtype=np.float64), np.zeros((3,), dtype=np.float64)
        return self._state.x[0:3].copy(), self._state.x[3:6].copy()


class CubePoseFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("cube_pose_fusion")

        # ===== Parameters =====
        self.declare_parameter("input.pose_topic", "/cube_pose/pose")
        self.declare_parameter("imu.topic", "/imu/data")
        self.declare_parameter("output.pose_topic", "/cube_pose/fused_pose")

        self.declare_parameter("fuse.position.enable", True)
        self.declare_parameter("fuse.position.use_imu_accel", False)
        self.declare_parameter("fuse.position.sigma_meas_m", 0.02)
        self.declare_parameter("fuse.position.sigma_accel_mps2", 2.0)

        self.declare_parameter("fuse.orientation.enable", True)
        # imu_weight=1 -> trust IMU entirely; imu_weight=0 -> trust vision entirely.
        self.declare_parameter("fuse.orientation.imu_weight", 0.9)
        self.declare_parameter("fuse.orientation.warn_on_frame_mismatch", True)

        # Optional extra smoothing on output position stream (simple window average).
        self.declare_parameter("fuse.position.output_smooth_window", 1)

        self._pose_topic = str(self.get_parameter("input.pose_topic").value)
        self._imu_topic = str(self.get_parameter("imu.topic").value)
        self._out_topic = str(self.get_parameter("output.pose_topic").value)

        self._enable_pos = bool(self.get_parameter("fuse.position.enable").value)
        self._use_imu_accel = bool(self.get_parameter("fuse.position.use_imu_accel").value)
        self._sigma_meas_m = float(self.get_parameter("fuse.position.sigma_meas_m").value)
        self._sigma_accel = float(self.get_parameter("fuse.position.sigma_accel_mps2").value)

        self._enable_ori = bool(self.get_parameter("fuse.orientation.enable").value)
        self._imu_weight = float(self.get_parameter("fuse.orientation.imu_weight").value)
        self._warn_frame_mismatch = bool(
            self.get_parameter("fuse.orientation.warn_on_frame_mismatch").value
        )

        self._out_pos_smooth_window = max(
            1, int(self.get_parameter("fuse.position.output_smooth_window").value)
        )
        self._pos_window: Deque[np.ndarray] = deque(maxlen=self._out_pos_smooth_window)

        # ===== State =====
        self._kf = LinearKFPos3D(sigma_accel=self._sigma_accel, sigma_meas=self._sigma_meas_m)
        self._last_imu: Optional[Imu] = None
        self._last_imu_t: Optional[float] = None
        self._warned_frame = False

        # ===== ROS I/O =====
        self._sub_pose = self.create_subscription(PoseStamped, self._pose_topic, self._on_pose, 10)
        self._sub_imu = self.create_subscription(Imu, self._imu_topic, self._on_imu, 50)
        self._pub = self.create_publisher(PoseStamped, self._out_topic, 10)

        self.get_logger().info(
            f"cube_pose_fusion started. pose_topic={self._pose_topic}, imu_topic={self._imu_topic}, out={self._out_topic}"
        )

    def _on_imu(self, msg: Imu) -> None:
        self._last_imu = msg
        t = _stamp_to_sec(msg.header.stamp)

        if self._last_imu_t is None:
            self._last_imu_t = t
            return

        if not self._enable_pos or not self._kf.initialized:
            self._last_imu_t = t
            return

        accel = None
        if self._use_imu_accel:
            accel = np.array(
                [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
                dtype=np.float64,
            )
        # Predict to IMU time.
        self._kf.predict(t_sec=t, accel_mps2=accel)
        self._last_imu_t = t

    def _on_pose(self, msg: PoseStamped) -> None:
        t = _stamp_to_sec(msg.header.stamp)

        # ===== Position fusion =====
        pos_meas = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float64
        )

        if self._enable_pos:
            if not self._kf.initialized:
                self._kf.initialize(pos_meas, t_sec=t)
            else:
                # Predict to vision time using last known IMU accel if enabled (otherwise constant-velocity)
                accel = None
                if self._use_imu_accel and self._last_imu is not None:
                    accel = np.array(
                        [
                            self._last_imu.linear_acceleration.x,
                            self._last_imu.linear_acceleration.y,
                            self._last_imu.linear_acceleration.z,
                        ],
                        dtype=np.float64,
                    )
                self._kf.predict(t_sec=t, accel_mps2=accel)
                self._kf.update_pos(pos_meas, t_sec=t)

            pos_fused, _vel = self._kf.get_pos_vel()
        else:
            pos_fused = pos_meas

        # Optional output smoothing on position stream (simple window average)
        self._pos_window.append(pos_fused.astype(np.float64))
        pos_out = np.mean(np.stack(list(self._pos_window), axis=0), axis=0)

        # ===== Orientation fusion =====
        q_vis = np.array(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float64,
        )
        q_out = q_vis
        if self._enable_ori and self._last_imu is not None:
            q_imu = np.array(
                [
                    self._last_imu.orientation.x,
                    self._last_imu.orientation.y,
                    self._last_imu.orientation.z,
                    self._last_imu.orientation.w,
                ],
                dtype=np.float64,
            )
            # If IMU doesn't provide orientation (all zeros), skip.
            if float(np.linalg.norm(q_imu)) > 1e-6:
                if (
                    self._warn_frame_mismatch
                    and not self._warned_frame
                    and self._last_imu.header.frame_id
                    and msg.header.frame_id
                    and self._last_imu.header.frame_id != msg.header.frame_id
                ):
                    self._warned_frame = True
                    self.get_logger().warn(
                        "IMU frame_id differs from pose frame_id. "
                        f"imu='{self._last_imu.header.frame_id}' pose='{msg.header.frame_id}'. "
                        "Orientation fusion assumes same frame; consider aligning frames or disabling orientation fusion."
                    )

                w = float(np.clip(self._imu_weight, 0.0, 1.0))
                q_out = _quat_slerp(q_vis, q_imu, w)

        # ===== Publish =====
        out = PoseStamped()
        out.header = msg.header
        out.pose.position.x = float(pos_out[0])
        out.pose.position.y = float(pos_out[1])
        out.pose.position.z = float(pos_out[2])
        out.pose.orientation.x = float(q_out[0])
        out.pose.orientation.y = float(q_out[1])
        out.pose.orientation.z = float(q_out[2])
        out.pose.orientation.w = float(q_out[3])
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CubePoseFusionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

