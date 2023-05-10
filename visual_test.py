from mmdet3d.apis import init_model, inference_detector, show_result_meshlab

config_file = 'configs/diffusionbev/diffusionbev_kitti_L.py'
checkpoint_file = '~/log/baseline/latest.pth'

# config_file = 'configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py'
# checkpoint_file = 'work_dirs/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/epoch_1.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')
pcd = '/mnt/data/ydy/dataset/kitti/testing/velodyne/000000.bin'
result, data = inference_detector(model, pcd)
out_dir = './out/baseline/'
show_result_meshlab(data, result, out_dir, show=False)