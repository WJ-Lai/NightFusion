from nuscenes.nuscenes import NuScenes

root = '/home/vincent/Data/v1.0-mini'
# root = r'D:\dataset\v1.0-mini'
nusc = NuScenes(version='v1.0-mini', dataroot=root, verbose=True)
my_sample = nusc.sample[50]
# nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP',
#                                 out_path='LIDAR_TOP.png')

# nusc.render_sample_data(my_sample['data']['CAM_FRONT'],out_path='CAM_FRONT.png')

# nusc.render_pointcloud_in_DHR(my_sample['token'], encode_type='depth', nsweeps=10,
#                               out_path='depth.png')
# nusc.render_pointcloud_in_DHR(my_sample['token'], encode_type='height', nsweeps=10,
#                               out_path='height.png')
# nusc.render_pointcloud_in_DHR(my_sample['token'], encode_type='intensity', nsweeps=10,
#                               out_path='intensity.png')
#
# nusc.render_pointcloud_in_image(nusc.sample[0]['token'])

# import cv2
# img = cv2.imread("depth.png")
# cv2.imshow('before',img)
# k=cv2.waitKey(0)
# img = img[0:900, 350:1250]  # 裁剪坐标为[y0:y1, x0:x1]
# cv2.imshow('after',img)
# k=cv2.waitKey(0)
import cv2
from nuscenes.image_utils import ImageProcessing as ip
im = ip('e3d495d4ac534d54b321f50006683844',nusc,416)
cv2.imwrite('cut.jpg', im.image)