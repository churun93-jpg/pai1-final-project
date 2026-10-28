# Physical AI 応用1講座 最終課題：AI視覚推論（YOLOv8）とLiDARを統合した自律移動ロボット制御

## 1. 概要
本プロジェクトは、**ROS 2 (Humble)** と **Gazebo** シミュレータを用い、差動二輪ロボット（TurtleBot3 Waffle Pi）に**事前学習済みAIモデル（YOLOv8）によるリアルタイム画像推論**と**2D LiDARセンサによる障害物回避・自律探索**を統合したシステムです。

課題サンプルの初級ベースライン（ROS 2とGazeboの連携によるロボット操作）を発展させ、単なる手動操作や固定ルール制御ではなく、**ロボット搭載カメラ映像に対するディープラーニング物体検出（YOLOv8n）の推論結果と距離センサ情報を統合して行動決定を行う自律パトロールタスク**を実装しました。

---

## 2. システム構成・特徴
- **シミュレータ環境**: Gazebo Classic + ROS 2 Humble (`turtlebot3_gazebo`)
- **使用ロボット**: TurtleBot3 `waffle_pi`（RGBカメラ `/camera/image_raw` および 360° LiDAR `/scan` 搭載）
- **AI推論モジュール**:
  - **YOLOv8n (`ultralytics`)** を用いたリアルタイム視覚推論（物体検出・関心領域認識）
  - カメラ映像から前方のターゲットおよび環境特徴を推論し、推論結果（検出クラス・信頼度・推論速度FPS）をリアルタイムHUD画面に可視化
- **自律行動決定ステートマシン**:
  1. `PATROL (自律探索)`: 前方安全確認を行いながら環境内を自律走行
  2. `TARGET_APPROACH (AIターゲット接近)`: AI推論によりターゲットを認識した際、対象方向へ旋回・接近してインスペクションを実行
  3. `OBSTACLE_AVOIDANCE (障害物回避)`: LiDARセンサで近距離障害物を検知した際、左右の空きスペースを比較して安全方向へ自動回避

---

## 3. ファイル構成
```text
pai_final_project/
├── README.md              # 本説明ファイル
├── ai_patrol_node.py      # AI推論（YOLOv8）＋LiDAR自律制御 ROS 2ノード
└── run_simulation.sh      # シミュレータ＆AIノード一括起動スクリプト
```

---

## 4. 実行方法

### 前提環境
- Ubuntu 22.04 LTS (WSL2 対応)
- ROS 2 Humble
- Gazebo (`ros-humble-gazebo-ros-pkgs`, `ros-humble-turtlebot3-simulations`)
- Pythonパッケージ: `ultralytics`, `opencv-python`, `numpy<2`

### ワンコマンド起動手順
```bash
chmod +x run_simulation.sh
./run_simulation.sh
```

または個別ターミナルで起動する場合：

**ターミナル1（Gazeboシミュレーション起動）:**
```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

**ターミナル2（AI自律制御ノード起動）:**
```bash
source /opt/ros/humble/setup.bash
python3 ai_patrol_node.py
```
