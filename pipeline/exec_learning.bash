#!/bin/sh

bag_name=rcup_20220218_pick
config_file=rarm_pr2.yaml
z_dim=10
c_epoch=100
hidden=256
p_epoch=20000
data_name=${bag_name}_z${z_dim}_e${c_epoch}
comp_model_name=${bag_name}_DCAE_z${z_dim}_e${c_epoch}
pred_model_name=${comp_model_name}_LSTM_noise_series_batch_e${p_epoch}_h${hidden}

base_path=$(dirname "$(realpath $0)")
scripts_path=$base_path/../scripts

# python3 ${scripts_path}/rosbag_convert_to_data.py -c ../configs/${config_file} -b ../bags/${bag_name}/ -d ${scripts_path}/data/${data_name}/
# python3 ${scripts_path}/train_DCAE.py -e ${c_epoch} -z ${z_dim} -d ${scripts_path}/data/${data_name}/ -m ../models/${comp_model_name}/
# python3 ${scripts_path}/test_DCAE.py -z ${z_dim} -d ${scripts_path}/data/${data_name}/ -m ../models/${comp_model_name}/
# python3 ${scripts_path}/comp_by_DCAE.py -z ${z_dim} -d ${scripts_path}/data/${data_name}/ -m ../models/${comp_model_name}/
python3 ${scripts_path}/train_LSTM.py -e ${p_epoch} -z ${z_dim} -h ${hidden} -d ${scripts_path}/data/${data_name}/ -m ../models/${pred_model_name}

mkdir_one_line=${scripts_path}/movies
if [[ -d "$mkdir_one_line" ]]; then
    echo "既に"$mkdir_one_line"は存在しています"
else
    mkdir "$mkdir_one_line"
    echo ""$mkdir_one_line"を作りました"
fi
mkdir_one_line=${scripts_path}/movies/${pred_model_name}
if [[ -d "$mkdir_one_line" ]]; then
    echo "既に"$mkdir_one_line"は存在しています"
else
    mkdir "$mkdir_one_line"
    echo ""$mkdir_one_line"を作りました"
fi

for file in `\find ${scripts_path}/data/${data_name}/ -name '*.bag'`; do
	echo "$file"
    file_name=${file##*/}
    before_period=${file_name%%.*}
    name=${before_period##*/}
    echo "$name"
    python3 ${scripts_path}/test_LSTM.py -l 0 -h ${hidden} -d ${file}/ -m ../models/${comp_model_name} -p ../models/${pred_model_name} -fp ${pred_model_name}/${pred_model_name}_${name} -z ${z_dim}
done
