#/bin/bash

project_name="cap_hook_20220331"
config_name="larm_gripper_pr2.yaml"
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
bag_zip="$bagdir/bag.zip"

if [[ ! -e $bagdir ]]; then
    mkdir -p $bagdir
fi

# temporary bag_zip file located in Kanazawanaoaki's google drive
if [[ ! -e $bag_zip ]]; then
    gdown https://drive.google.com/uc?id=1t5aN4ZZztOkUMHY3LWWwPU-5LF_ZBipR -O $bag_zip
    unzip $bag_zip -d $bagdir
fi

rosrun jsk_learning_utils rosbag_convert_to_data.py -project $project_name -config $config_name
rosrun jsk_learning_utils train_DCAE.py -z $zdim -e $c_epoch -project $project_name
rosrun jsk_learning_utils test_DCAE.py -z $zdim -project $project_name
rosrun jsk_learning_utils comp_by_DCAE.py -z $zdim -project $project_name
rosrun jsk_learning_utils train_LSTM.py -z $zdim -hidden $hidden -e $p_epoch -project $project_name
rosrun jsk_learning_utils test_LSTM.py -z $zdim -l 0 -hidden $hidden -project $project_name
