from nuscenes.nuscenes import NuScenes
from nuscenes.nuscenes import NuScenesExplorer
# import tensorflow as tf
import cv2
import numpy as np
from nuscenes.image_utils import ImageProcessing as ip


root = 'D:/dataset/v1.0-mini'
# root = '/home/vincent/Data/v1.0-mini'
nusc = NuScenes(version='v1.0-mini', dataroot=root, verbose=True)


# get night scenes
night_scene = nusc.list_night()

# Get sensor data
sensor_token = nusc.get_sensor_token_list(night_scene[0])
# cv2读取图片颜色发生变化
# sensor_token = nusc.get_sensor_token_list('scene-0061')
im = ip(sensor_token[0], nusc, min_size=900)

# Render ground truth
# im.render_arbitrary()
# im.render_arbitrary1()
# Render prediction
# im.render_rectangle(im.control_points_list)
# cell_class, corner_point_deta= im.annotations_in_net()

im = ip('2d35f62a9daf41a18c15c7a946b0d141', nusc, min_size=900)
im.render_arbitrary()
print(len(im.control_points_list))
print(im.control_points_list)

# my_sample = nusc.sample[300]
# im = ip(my_sample['data']['CAM_FRONT'], nusc, min_size=900)
# print(len(im.control_points_list))
# nusc.render_sample_data(my_sample['data']['CAM_FRONT'])
# nusc.render_sample_data(my_sample['data']['LIDAR_TOP'])