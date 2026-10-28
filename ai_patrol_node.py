#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physical AI 応用1講座 最終課題
AI視覚推論（YOLOv8）と2D LiDARを統合した自律移動ロボット制御ノード

機能概要:
1. Gazebo上のTurtleBot3 (waffle_pi) からRGBカメラ映像 (/camera/image_raw) と LiDAR (/scan) を取得
2. Gazebo空間全体を俯瞰する3Dカメラ (/overhead_cam/image_raw) とAI認識用ランドマークを動的スポーン
3. 事前学習済みディープラーニングモデル YOLOv8n (ultralytics) によりリアルタイム物体検出・視覚推論を実行
4. AI視覚推論結果とLiDAR距離センサ情報を統合し、以下の3つの状態を自律的に切り替えながら走行:
   - PATROL (自律探索走行)
   - AI_TARGET_APPROACH (AI認識対象への接近・インスペクション)
   - OBSTACLE_AVOIDANCE (LiDARによる障害物回避)
5. 「Gazebo 3D俯瞰映像」「ロボット搭載カメラAI推論映像」「2D LiDARレーダー」を統合したHDダッシュボードを表示・録画
"""

import math
import time
import os
import argparse
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SpawnEntity
from cv_bridge import CvBridge
import cv2
import numpy as np

# YOLOv8 (Ultralytics) の読み込み
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Gazeboワールド内に配置する視覚ターゲット（赤・緑・青のランドマーク）のSDFテンプレート
TARGET_SDF_TEMPLATE = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <pose>0 0 0.25 0 0 0</pose>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.12</radius>
            <length>0.5</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{r} {g} {b} 1</ambient>
          <diffuse>{r} {g} {b} 1</diffuse>
          <emissive>{r} {g} {b} 0.3</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

# Gazebo空間全体を撮影する俯瞰3DカメラのSDFテンプレート
OVERHEAD_CAM_SDF = """<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="overhead_camera_model">
    <static>true</static>
    <link name="link">
      <pose>-2.4 -2.4 3.3 0 0.72 0.785</pose>
      <sensor name="overhead_cam" type="camera">
        <always_on>true</always_on>
        <update_rate>15.0</update_rate>
        <camera>
          <horizontal_fov>1.15</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>30.0</far>
          </clip>
        </camera>
        <plugin name="overhead_cam_controller" filename="libgazebo_ros_camera.so">
          <ros>
            <namespace>overhead</namespace>
          </ros>
          <camera_name>cam</camera_name>
          <frame_name>overhead_cam_link</frame_name>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>
"""


class AIPatrolNode(Node):
    def __init__(self, max_duration: float = 0.0):
        super().__init__('ai_patrol_node')
        self.get_logger().info('=== AI Patrol Node (YOLOv8 + LiDAR) を初期化中 ===')

        self.max_duration = max_duration
        self.bridge = CvBridge()
        self.latest_image = None
        self.overhead_image = None
        self.latest_scan = None

        # LiDAR距離特徴量 (前方・左前方・右前方)
        self.front_dist = 10.0
        self.left_dist = 10.0
        self.right_dist = 10.0
        self.scan_ranges = []

        # AIモデル (YOLOv8n) のロード
        self.yolo_model = None
        if YOLO_AVAILABLE:
            self.get_logger().info('YOLOv8n 事前学習済みモデルをロードしています...')
            self.yolo_model = YOLO('yolov8n.pt')
            self.get_logger().info('YOLOv8n モデルのロードが完了しました。')
        else:
            self.get_logger().warn('ultralytics が見つからないため、OpenCV視覚推論モードで動作します。')

        # QoS設定
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        # サブスクライバ & パブリッシャ
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.overhead_sub = self.create_subscription(
            Image, '/overhead/cam/image_raw', self.overhead_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, sensor_qos
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # 状態管理
        self.state = 'PATROL'
        self.state_detail = 'Searching environment with YOLOv8 & LiDAR'
        self.start_time = None
        self.recorded_frames = 0
        self.turn_direction = 1.0
        self.inspected_targets = set()
        self.inspect_cooldown_until = 0.0

        # 推論結果の保持
        self.ai_detections = []
        self.inference_ms = 0.0
        self.target_offset_x = 0.0
        self.target_detected = False
        self.target_label = ''

        # 動画保存用ライター (1280x720 HD, 10fps)
        self.video_writer = None
        self.output_video_path = os.path.join(os.getcwd(), 'ai_patrol_raw.mp4')
        self.finished = False

        # Gazebo内に視覚ランドマーク＆俯瞰カメラをスポーン
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.targets_spawned = False
        self.spawn_timer = self.create_timer(2.0, self.spawn_entities_once)

        # メイン制御＆AI推論ループ (10Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('=== AI自律制御ループを開始しました (10 Hz) ===')

    def spawn_entities_once(self):
        """Gazebo空間内に俯瞰カメラとAI認識用ランドマークオブジェクトを配置する"""
        if self.targets_spawned:
            return
        if not self.spawn_client.service_is_ready():
            return

        self.targets_spawned = True

        # 0. ロボット本体 (waffle_pi) が未スポーンの場合に備えて確実にスポーン
        tb3_sdf_path = '/opt/ros/humble/share/turtlebot3_gazebo/models/turtlebot3_waffle_pi/model.sdf'
        if os.path.exists(tb3_sdf_path):
            with open(tb3_sdf_path, 'r', encoding='utf-8') as f:
                tb3_req = SpawnEntity.Request()
                tb3_req.name = 'waffle_pi'
                tb3_req.xml = f.read()
                tb3_req.initial_pose.position.x = -2.0
                tb3_req.initial_pose.position.y = -0.5
                tb3_req.initial_pose.position.z = 0.01
                self.spawn_client.call_async(tb3_req)

        # 1. 俯瞰3Dカメラの配置
        cam_req = SpawnEntity.Request()
        cam_req.name = 'overhead_camera_model'
        cam_req.xml = OVERHEAD_CAM_SDF
        self.spawn_client.call_async(cam_req)

        # 2. 視覚ターゲット3基の配置
        landmarks = [
            ('ai_beacon_red', 1.6, 0.6, 1.0, 0.1, 0.1),
            ('ai_beacon_green', 0.5, -1.6, 0.1, 0.9, 0.2),
            ('ai_beacon_blue', -1.5, 0.8, 0.1, 0.4, 1.0),
        ]
        for name, x, y, r, g, b in landmarks:
            req = SpawnEntity.Request()
            req.name = name
            req.xml = TARGET_SDF_TEMPLATE.format(name=name, r=r, g=g, b=b)
            req.initial_pose.position.x = float(x)
            req.initial_pose.position.y = float(y)
            req.initial_pose.position.z = 0.0
            self.spawn_client.call_async(req)
        self.get_logger().info('Gazebo空間に俯瞰3DカメラとAI認識用ランドマーク3基を配置しました。')

    def image_callback(self, msg: Image):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'画像変換エラー: {e}')

    def overhead_callback(self, msg: Image):
        try:
            self.overhead_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            pass

    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isinf(ranges) | np.isnan(ranges) | (ranges <= 0.01), 10.0, ranges)
        self.scan_ranges = ranges.tolist()

        n = len(ranges)
        if n == 0:
            return

        front_indices = list(range(0, int(n * 28 / 360))) + list(range(int(n * 332 / 360), n))
        left_indices = list(range(int(n * 28 / 360), int(n * 80 / 360)))
        right_indices = list(range(int(n * 280 / 360), int(n * 332 / 360)))

        self.front_dist = float(np.min(ranges[front_indices])) if front_indices else 10.0
        self.left_dist = float(np.min(ranges[left_indices])) if left_indices else 10.0
        self.right_dist = float(np.min(ranges[right_indices])) if right_indices else 10.0

    def run_ai_inference(self, frame: np.ndarray):
        """YOLOv8推論および視覚ターゲット認識を実行し、推論結果リストを返す"""
        t0 = time.perf_counter()
        detections = []
        h, w, _ = frame.shape

        # 1. YOLOv8 ディープラーニング推論
        if self.yolo_model is not None:
            results = self.yolo_model.predict(frame, imgsz=320, conf=0.12, verbose=False)
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0].item())
                    cls_id = int(box.cls[0].item())
                    cls_name = self.yolo_model.names.get(cls_id, f'class_{cls_id}')
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'label': f'YOLOv8:{cls_name}',
                        'conf': conf,
                        'color': (0, 255, 255),
                        'cx': (x1 + x2) / 2.0,
                        'area': (x2 - x1) * (y2 - y1)
                    })

        # 2. 視覚特徴（ランドマーク領域）の認識
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_specs = [
            ('Beacon-RED', np.array([0, 140, 90]), np.array([10, 255, 255]), (60, 60, 255)),
            ('Beacon-GREEN', np.array([40, 120, 80]), np.array([85, 255, 255]), (60, 255, 60)),
            ('Beacon-BLUE', np.array([100, 130, 80]), np.array([135, 255, 255]), (255, 160, 50)),
        ]
        for label, lower, upper, bgr_color in color_specs:
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > (w * h * 0.0025):
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    conf = min(0.99, 0.72 + (area / (w * h)) * 3.0)
                    detections.append({
                        'bbox': (x, y, x + bw, y + bh),
                        'label': f'AI-Target:{label}',
                        'conf': conf,
                        'color': bgr_color,
                        'cx': x + bw / 2.0,
                        'area': area
                    })

        self.inference_ms = (time.perf_counter() - t0) * 1000.0
        self.ai_detections = detections

        self.target_detected = False
        self.target_offset_x = 0.0
        self.target_label = ''

        now = time.time()
        if detections and now > self.inspect_cooldown_until:
            best = max(detections, key=lambda d: d['area'])
            self.target_offset_x = (best['cx'] - (w / 2.0)) / (w / 2.0)
            self.target_detected = True
            self.target_label = f"{best['label']} ({best['conf']*100:.0f}%)"

            if best['area'] > (w * h * 0.075):
                self.inspected_targets.add(best['label'])
                self.inspect_cooldown_until = now + 4.5
                self.get_logger().info(f"[AI完了] {best['label']} の接近確認を完了！次の探索へ移行します。")

    def control_loop(self):
        """AI視覚推論＋LiDARに基づく行動決定ループ"""
        if self.finished:
            return

        # カメラ画像が届いてから録画・タイマーを開始
        if self.latest_image is None:
            return

        if self.start_time is None:
            self.start_time = time.time()
            self.get_logger().info('カメラ映像およびセンサ受信を確認！録画とAI走行を開始します。')

        self.run_ai_inference(self.latest_image)

        cmd = Twist()

        # 優先度1: LiDARによる近距離障害物回避 (安全保証レイヤー)
        if self.front_dist < 0.58:
            self.state = 'OBSTACLE_AVOIDANCE'
            if self.left_dist >= self.right_dist:
                self.turn_direction = 1.0
                dir_str = 'LEFT'
            else:
                self.turn_direction = -1.0
                dir_str = 'RIGHT'
            cmd.linear.x = 0.0
            cmd.angular.z = 0.70 * self.turn_direction
            self.state_detail = f'LiDAR Obstacle at {self.front_dist:.2f}m -> Turn {dir_str}'

        # 優先度2: AIが視覚ターゲットを検出した場合はターゲット方向へ制御
        elif self.target_detected:
            self.state = 'AI_TARGET_APPROACH'
            cmd.linear.x = 0.17
            cmd.angular.z = float(np.clip(-0.85 * self.target_offset_x, -0.65, 0.65))
            self.state_detail = f'Approaching {self.target_label} (err={self.target_offset_x:+.2f})'

        # 優先度3: 通常パトロール
        else:
            self.state = 'PATROL'
            cmd.linear.x = 0.21
            wall_balance = np.clip((self.left_dist - self.right_dist) * 0.18, -0.40, 0.40)
            cmd.angular.z = float(wall_balance)
            self.state_detail = f'Autonomous Patrol (Front Clear: {self.front_dist:.2f}m)'

        self.cmd_pub.publish(cmd)
        self.render_dashboard(cmd)

        # 指定秒数(例: 75秒)に達したら自動停止
        elapsed = self.recorded_frames / 10.0
        if self.max_duration > 0 and elapsed >= self.max_duration:
            self.get_logger().info(f'指定時間 ({self.max_duration}秒) の録画が完了しました。終了処理を行います。')
            self.finished = True
            self.cmd_pub.publish(Twist())
            raise SystemExit(0)

    def render_dashboard(self, cmd: Twist):
        """1280x720 HD統合ダッシュボードを描画・録画"""
        canvas = np.full((720, 1280, 3), 22, dtype=np.uint8)

        # 上部ヘッダーバー (高さ60px)
        cv2.rectangle(canvas, (0, 0), (1280, 58), (38, 42, 52), -1)
        cv2.putText(canvas, 'Physical AI Final Project: YOLOv8 Vision & 2D LiDAR Autonomous Robot (ROS 2 + Gazebo)',
                    (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # 左パネル: Gazebo 3D Simulation View (620x630)
        cv2.putText(canvas, '[1] Gazebo 3D Simulation Environment (Overhead View)',
                    (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (120, 220, 255), 2)
        if self.overhead_image is not None:
            ov_view = cv2.resize(self.overhead_image, (610, 590))
        else:
            ov_view = cv2.resize(self.latest_image, (610, 590))
        canvas[102:692, 20:630] = ov_view
        cv2.rectangle(canvas, (20, 102), (630, 692), (90, 90, 90), 2)

        # 右上パネル: Robot Onboard Camera + YOLOv8 Inference (610x340)
        cv2.putText(canvas, '[2] Robot Onboard Camera (/camera/image_raw) + YOLOv8 Inference',
                    (650, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 160), 2)
        cam_w, cam_h = 610, 330
        cam_view = cv2.resize(self.latest_image, (cam_w, cam_h))
        orig_h, orig_w = self.latest_image.shape[:2]
        sx, sy = cam_w / float(orig_w), cam_h / float(orig_h)

        for det in self.ai_detections:
            x1, y1, x2, y2 = det['bbox']
            rx1, ry1 = int(x1 * sx), int(y1 * sy)
            rx2, ry2 = int(x2 * sx), int(y2 * sy)
            color = det['color']
            cv2.rectangle(cam_view, (rx1, ry1), (rx2, ry2), color, 2)
            text = f"{det['label']} {det['conf']*100:.0f}%"
            cv2.rectangle(cam_view, (rx1, max(0, ry1 - 22)), (rx1 + len(text) * 9, ry1), color, -1)
            cv2.putText(cam_view, text, (rx1 + 3, max(15, ry1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2)

        cv2.rectangle(cam_view, (8, 8), (380, 64), (20, 20, 20), -1)
        cv2.putText(cam_view, 'AI Engine: Ultralytics YOLOv8n + HSV Detector', (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        cv2.putText(cam_view, f'Inference Latency: {self.inference_ms:.1f} ms | Detections: {len(self.ai_detections)}',
                    (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 255, 180), 1)

        canvas[102:432, 650:1260] = cam_view
        cv2.rectangle(canvas, (650, 102), (1260, 432), (90, 90, 90), 2)

        # 右下パネル: 2D LiDAR Radar & Autonomous Control Telemetry (610x240)
        cv2.rectangle(canvas, (650, 450), (1260, 692), (32, 35, 42), -1)
        cv2.rectangle(canvas, (650, 450), (1260, 692), (90, 90, 90), 2)

        # レーダー描画 (右側エリア)
        cx, cy = 1135, 575
        max_r = 95
        cv2.circle(canvas, (cx, cy), max_r, (80, 80, 80), 1)
        cv2.circle(canvas, (cx, cy), max_r // 2, (55, 55, 55), 1)
        cv2.line(canvas, (cx, cy - max_r), (cx, cy + max_r), (50, 50, 50), 1)
        cv2.line(canvas, (cx - max_r, cy), (cx + max_r, cy), (50, 50, 50), 1)

        if self.scan_ranges:
            n = len(self.scan_ranges)
            for i, r in enumerate(self.scan_ranges):
                if r < 3.5:
                    angle = (2.0 * math.pi * i) / n
                    px = int(cx - (r / 3.5) * max_r * math.sin(angle))
                    py = int(cy - (r / 3.5) * max_r * math.cos(angle))
                    pt_color = (60, 80, 255) if r < 0.58 else (80, 240, 120)
                    cv2.circle(canvas, (px, py), 2, pt_color, -1)

        cv2.circle(canvas, (cx, cy), 5, (0, 220, 255), -1)
        cv2.putText(canvas, '2D LiDAR (/scan)', (1065, 472),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        # テキストテレメトリ
        state_color = (80, 255, 120)
        if self.state == 'OBSTACLE_AVOIDANCE':
            state_color = (80, 110, 255)
        elif self.state == 'AI_TARGET_APPROACH':
            state_color = (0, 230, 255)

        elapsed_sec = self.recorded_frames / 10.0
        cv2.putText(canvas, '[3] AI Decision & Sensor Fusion Telemetry', (668, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 120), 2)
        cv2.putText(canvas, f'MODE : {self.state}', (668, 518),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, state_color, 2)
        cv2.putText(canvas, f'DETAIL: {self.state_detail[:46]}', (668, 550),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
        cv2.putText(canvas, f'LiDAR : Front={self.front_dist:.2f}m | Left={self.left_dist:.2f}m | Right={self.right_dist:.2f}m',
                    (668, 582), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 230, 255), 1)
        cv2.putText(canvas, f'CMD_VEL: linear.x = {cmd.linear.x:.2f} m/s | angular.z = {cmd.angular.z:+.2f} rad/s',
                    (668, 614), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 230, 255), 1)
        cv2.putText(canvas, f'Simulation Time: {elapsed_sec:05.1f} s  |  Inspected Landmarks: {len(self.inspected_targets)}',
                    (668, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 200), 1)

        # 動画書き込み (10fps)
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, 10.0, (1280, 720))
        self.video_writer.write(canvas)
        self.recorded_frames += 1

        try:
            cv2.imshow('Physical AI - Autonomous Robot Dashboard', canvas)
            cv2.waitKey(1)
        except Exception:
            pass

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=0.0, help='Recording duration in seconds (0 for infinite)')
    parsed_args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = AIPatrolNode(max_duration=parsed_args.duration)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info('ノードを正常終了します...')
    finally:
        stop_msg = Twist()
        node.cmd_pub.publish(stop_msg)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
