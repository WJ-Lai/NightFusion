from nuscenes.nuscenes import NuScenes

root = '/home/vincent/Data/v1.0-mini'
root = r'D:\dataset\v1.0-mini'
nusc = NuScenes(version='v1.0-mini', dataroot=root, verbose=True)
my_sample = nusc.sample[50]
nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP',
                                out_path='LIDAR_TOP.png')

# nusc.render_sample_data(my_sample['data']['CAM_FRONT'],out_path='CAM_FRONT.png')

nusc.render_pointcloud_in_DHR(my_sample['token'], encode_type='depth', nsweeps=10,
                              out_path='depth.png')
nusc.render_pointcloud_in_DHR(my_sample['token'], encode_type='height', nsweeps=10,
                              out_path='height.png')
nusc.render_pointcloud_in_DHR(my_sample['token'], encode_type='intensity', nsweeps=10,
                              out_path='intensity.png')

nusc.render_pointcloud_in_image(nusc.sample[0]['token'])