import os
import rospkg

def create_dir_if_not_exist(dirname: str) -> None:
    if not os.path.exists(dirname):
        os.mkdir(dirname)

def get_project_dir(project_name: str) -> str:
    pkg_path = rospkg.RosPack().get_path('jsk_learning_utils')
    return os.path.join(pkg_path, 'project_data', project_name)

def get_rosbag_dir(project_name: str) -> str:
    dirname = os.path.join(get_project_dir(project_name), 'bags')
    assert os.path.exists(dirname), '{} has not been created yet'.format(dirname)
    return dirname

def get_dataset_dir(project_name: str) -> str:
    dirname = os.path.join(get_project_dir(project_name), 'dataset')
    create_dir_if_not_exist(dirname)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    return dirname

def get_kanazawa_specific_rosbag_dir(project_name: str, rosbag_file_name: str) -> str:
    # TODO: remove this later
    kanazawa_rosbag_dir = os.path.join(get_dataset_dir(project_name), rosbag_file_name)
    create_dir_if_not_exist(kanazawa_rosbag_dir)
    return kanazawa_rosbag_dir
