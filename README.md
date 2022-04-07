# jsk_learning_utils
JSKのロボットで模倣学習等を行うための関連プログラム．  
基本的にPython3を使う．  
ロボットでのデータ収集関連とrosbagからのデータ変換，学習したモデルを実行するためのROSプログラムなど．

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