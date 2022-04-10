#/bin/bash

project_name="sample_rcup_pick"
zdim=10
testing=false
if $testing
then
    c_epoch=2
    p_epoch=2
else
    c_epoch=100
    p_epoch=20000
fi
hidden=256

project_dir=$(rospack find jsk_learning_utils)/project_data/$project_name
bagdir="$project_dir/bags"
bag_zip="$bagdir/bag.zip" # temporary located in HiroIshida's google drive

if [[ ! -e $bagdir ]]; then
    mkdir -p $bagdir
fi

if [[ ! -e $bag_zip ]]; then
    gdown https://drive.google.com/uc?id=1pd24iFEXf8PYkpPvu9OJg4ieV8cpj9rn -O $bag_zip
    unzip $bag_zip -d $bagdir
fi

rosrun jsk_learning_utils scripts/rosbag_convert_to_data.py
rosrun jsk_learning_utils train_DCAE.py -z $zdim -e $c_epoch -project sample_rcup_pick
rosrun jsk_learning_utils test_DCAE.py -z $zdim -project sample_rcup_pick
rosrun jsk_learning_utils comp_by_DCAE.py -z $zdim -project sample_rcup_pick
rosrun jsk_learning_utils train_LSTM.py -z $zdim -hidden $hidden -e $p_epoch -project sample_rcup_pick
rosrun jsk_learning_utils test_LSTM.py -z $zdim -l 0 -hidden $hidden -project sample_rcup_pick
