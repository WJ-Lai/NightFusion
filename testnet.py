from nuscenes.nuscenes import NuScenes

root = '/home/vincent/Data/v1.0-mini'
nusc = NuScenes(version='v1.0-mini', dataroot=root, verbose=True)
my_sample = nusc.sample[10]
nusc.render_pointcloud_in_image(my_sample['token'], pointsensor_channel='LIDAR_TOP',
                                out_path='LIDAR_TOP.png')

nusc.render_pointcloud_in_DHR(my_sample['token'], encode_type='depth',
                              out_path='depth.png')
nusc.render_pointcloud_in_DHR(my_sample['token'], encode_type='height',
                              out_path='height.png')
nusc.render_pointcloud_in_DHR(my_sample['token'], encode_type='reflectance',
                              out_path='reflectance.png')

nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], nsweeps=5)