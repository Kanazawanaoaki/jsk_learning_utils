# jsk_learning_utils
JSKのロボットで模倣学習等を行うための関連プログラム．  
基本的にPython3を使う．  
ロボットでのデータ収集関連とrosbagからのデータ変換，学習したモデルを実行するためのROSプログラムなど．

## Workspace build(melodic)
Build ROS with Python3 environment  
Environment:  Ubuntu18.04 and ROS Melodic
```
sudo apt-get install python3-catkin-pkg-modules python3-rospkg-modules python3-venv python3-empy
sudo apt-get install python3-opencv
sudo apt-get install ros-melodic-catkin
source /opt/ros/melodic/setup.bash
mkdir -p ~/learning_ws/src
cd ~/learning_ws/src
git clone https://github.com/Kanazawanaoaki/jsk_learning_utils.git
wstool init
wstool merge jsk_learning_utils/fc.rosinstall
wstool merge jsk_learning_utils/fc.rosinstall.melodic
wstool update
rosdep install --from-paths . --ignore-src -y -r
cd ~/learning_ws
catkin init
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
catkin build
```

## WIP 金沢のPR2の模倣学習
一時的に金沢の模倣学習の実行方法を書いておく．未完成だし，dataの保存方法（特にDCAEで圧縮したデータの保存方法）など諸々検討が必要．

### データ収集

#### ミラーでデータ収集
右腕をマネキンモードにして動かし，左腕に関節角度を送ってデータを集める．
```
roslaunch jsk_learning_utils data_collection_rosbag.launch
```

```
roscd jsk_learning_utils/euslisp/pr2
roseus pr2_mirror_inverse.l
```

#### 自動pick&placeデータ収集
一度の教示で自動pick&placeデータ収集
```

```

### データの学習
データの学習の一連のプログラムを走らせる．
`/bags`ディレクトリにある `exec_learning.bash`の`bag_name`と`config_file`などの情報を適宜変更して以下を実行する．
```
roscd jsk_learning_utils/pipeline
bash exec_learning.bash
```

### 学習した動作の実行
現在のカメラ画像と関節情報を取得して模倣学習の予測結果を出力するpythonプログラムを立ち上げる.
```
python3 imitation_ros_config.py -c ../configs/rarm_pr2.yaml -z 10 -l 50 -h 256 -m ../models/rcup_20220218_pick_DCAE_z10_e100 -p ../models/rcup_20220218_pick_DCAE_z10_e100_LSTM_noise_series_batch_e20000_h256
```
PR2に動作を送るeusプログラムを立ち上げる
```
roscd jsk_learning_utils/euslisp/pr2
roseus pr2_imitation_exec.l
(initial-pose-tmp)
```
模倣を実行する．
```
(start-imitation)
```

## WIP 石田さんの模倣学習への対応
金沢の模倣学習の`/scripts/data`のデータを石田さんのmohouのデータの形に変換する．  
石田さんの模倣学習のみをするなら，
```
roscd jsk_learning_utils/scripts
python3 rosbag_convert_to_data.py -c ../configs/rarm_pr2.yaml -b ../bags/rcup_20220218_pick -d data/rcup_20220218_pick
python3 convert_data_to_pickle_for_mohou.py -d data/rcup_20220218_pick -n rcup_20220218_pick
```
`-d`:`/scripts/data`以下のデータディレクトリ，`-n`:`~/.mohou`以下に作られるプロジェクトの名前．  

既に金沢模倣学習プログラムを実行している場合は`scripts/data`にデータディレクトリが作られているので，
```
roscd jsk_learning_utils/scripts
python3 convert_data_to_pickle_for_mohou.py -d data/rcup_20220218_pick_z10_e100 -n rcup_20220218_pick
```
