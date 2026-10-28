#!/bin/bash
set -e

source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
export GAZEBO_MODEL_DATABASE_URI=""
export GAZEBO_MODEL_PATH=/root/.gazebo/models:/opt/ros/humble/share/turtlebot3_gazebo/models:$GAZEBO_MODEL_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ffmpegが無ければインストール
if ! command -v ffmpeg >/dev/null 2>&1; then
    apt-get update && apt-get install -y ffmpeg
fi

# Gazeboがオンラインモデル取得で90秒フリーズするのを防ぐためローカルにsunとground_planeを用意
mkdir -p /root/.gazebo/models/sun /root/.gazebo/models/ground_plane

cat << 'EOF' > /root/.gazebo/models/sun/model.config
<?xml version="1.0"?>
<model>
  <name>Sun</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
</model>
EOF

cat << 'EOF' > /root/.gazebo/models/sun/model.sdf
<?xml version="1.0" ?>
<sdf version="1.6">
  <light type="directional" name="sun">
    <cast_shadows>true</cast_shadows>
    <pose>0 0 10 0 0 0</pose>
    <diffuse>0.8 0.8 0.8 1</diffuse>
    <specular>0.2 0.2 0.2 1</specular>
    <attenuation>
      <range>1000</range>
      <constant>0.9</constant>
      <linear>0.01</linear>
      <quadratic>0.001</quadratic>
    </attenuation>
    <direction>-0.5 0.1 -0.9</direction>
  </light>
</sdf>
EOF

cat << 'EOF' > /root/.gazebo/models/ground_plane/model.config
<?xml version="1.0"?>
<model>
  <name>Ground Plane</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
</model>
EOF

cat << 'EOF' > /root/.gazebo/models/ground_plane/model.sdf
<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="ground_plane">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>100 100</size>
          </plane>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>100</mu>
              <mu2>50</mu2>
            </ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <cast_shadows>false</cast_shadows>
        <geometry>
          <plane>
            <normal>0 0 1</normal>
            <size>100 100</size>
          </plane>
        </geometry>
        <material>
          <ambient>0.75 0.75 0.75 1</ambient>
          <diffuse>0.75 0.75 0.75 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>
EOF

# 既存プロセスのクリーンアップ
pkill -9 gzserver 2>/dev/null || true
pkill -9 gzclient 2>/dev/null || true
sleep 1

echo "=== [1/3] Gazeboシミュレーション (TurtleBot3 Waffle Pi) を起動しています... ==="
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py > /tmp/gazebo.log 2>&1 &
GAZEBO_PID=$!

echo "Gazeboの初期化とセンサ起動を待機しています (8秒)..."
sleep 8

echo "=== [2/3] AI自律制御ノード (YOLOv8 + LiDAR) を起動し、75秒間(1分15秒)の走行と録画を行います... ==="
python3 "$SCRIPT_DIR/ai_patrol_node.py" --duration 75

echo "=== [3/3] シミュレーション完了！動画を保存しています... ==="
kill $GAZEBO_PID 2>/dev/null || true
pkill -9 gzserver 2>/dev/null || true
pkill -9 gzclient 2>/dev/null || true

if command -v ffmpeg >/dev/null 2>&1 && [ -f "$SCRIPT_DIR/ai_patrol_raw.mp4" ]; then
    ffmpeg -y -i "$SCRIPT_DIR/ai_patrol_raw.mp4" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$SCRIPT_DIR/../提出用動画_PAI最終課題.mp4"
    cp "$SCRIPT_DIR/../提出用動画_PAI最終課題.mp4" "$SCRIPT_DIR/ai_patrol_demo.mp4"
    rm -f "$SCRIPT_DIR/ai_patrol_raw.mp4"
elif [ -f "$SCRIPT_DIR/ai_patrol_raw.mp4" ]; then
    cp "$SCRIPT_DIR/ai_patrol_raw.mp4" "$SCRIPT_DIR/../提出用動画_PAI最終課題.mp4"
    mv "$SCRIPT_DIR/ai_patrol_raw.mp4" "$SCRIPT_DIR/ai_patrol_demo.mp4"
fi

echo "=== 完了！提出用動画が作成されました: 提出用動画_PAI最終課題.mp4 ==="
