#/bin/bash
project_name="sample_rcup_pick"
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
