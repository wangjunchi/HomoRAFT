'''
Attention! The script generate plane matching between image pairs, make sure you alreday generated
the images pairs and had p.csv and q.csv stored in dataset folder.
'''

import cv2
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d
from hypersim_image_pairs import HypersimPairDataset
import pandas as pd


def main(scene_list):
    for scene in scene_list:
        process_scene(scene)

def process_scene(scene_name):
    dataset = HypersimPairDataset(data_dir="/cluster/project/infk/cvg/students/junwang/hypersim", scene_name=scene_name)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    # scene_name = dataset.scene_name

    csv_path = "/cluster/project/infk/cvg/students/junwang/hypersim/{}/homography.csv".format(scene_name)
    df = pd.DataFrame(columns=['scene_name', 'camera_id_1', 'frame_id_1', 'plane_id_1',
                               'camera_id_2', 'frame_id_2', 'plane_id_2',
                                'homography'])
    print("Processing scene {}".format(scene_name))
    for i, sample_pair in tqdm(enumerate(dataset)):
        # debug
        # if i > 5:
        #     break
        sample_1 = sample_pair['sample_1']
        sample_2 = sample_pair['sample_2']
        gt_match = sample_pair['gt_match']

        # check if two sample comes from the same scene
        scene_id_1 = sample_1['metadata'][0]
        scene_id_2 = sample_2['metadata'][0]
        assert scene_id_1 == scene_id_2, "Two samples should come from the same scene"

        camera_id_1 = sample_1['metadata'][1]
        camera_id_2 = sample_2['metadata'][1]
        frame_id_1 = str(sample_1['metadata'][2]).zfill(4)
        frame_id_2 = str(sample_2['metadata'][2]).zfill(4)

        # construct the projection matrix
        camera_pose_1 = sample_1['camera_pose']
        r1 = camera_pose_1[:3, :3]
        t = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]])

        r1 = np.matmul(t, r1)
        r1 = np.matmul(r1, np.linalg.inv(t))
        t1 = camera_pose_1[:3, 3]
        t1[1] = -t1[1]
        t1[2] = -t1[2]
        camera_pose_1 = np.concatenate([r1, t1.reshape(3, 1)], axis=1)
        camera_pose_1 = np.concatenate([camera_pose_1, np.array([[0, 0, 0, 1]])], axis=0)
        # convert from right hand to left hand


        # camera_pose_1 = np.linalg.inv(camera_pose_1)
        camera_pose_2 = sample_2['camera_pose']
        r2 = camera_pose_2[:3, :3]
        r2 = np.matmul(t, r2)
        r2 = np.matmul(r2, np.linalg.inv(t))
        t2 = camera_pose_2[:3, 3]
        t2[1] = -t2[1]
        t2[2] = -t2[2]
        camera_pose_2 = np.concatenate([r2, t2.reshape(3, 1)], axis=1)
        camera_pose_2 = np.concatenate([camera_pose_2, np.array([[0, 0, 0, 1]])], axis=0)

        # show image
        image_1 = sample_1['image'].numpy()
        image_1 = image_1.transpose(1, 2, 0)
        image_2 = sample_2['image'].numpy()
        image_2 = image_2.transpose(1, 2, 0)

        # plane segmentation
        planes_1 = sample_1['planes']
        plane_seg_1 = planes_1[:,:, 1]
        planes_2 = sample_2['planes']
        plane_seg_2 = planes_2[:,:, 1]

        # depth
        depth_1 = sample_1['depth']
        depth_2 = sample_2['depth']

        # image 2 as sample
        K = o3d.camera.PinholeCameraIntrinsic(1024, 768, 886.81, 886.81, 512, 384)
        depth_2 = o3d.geometry.Image(depth_2)
        pcd_2 = o3d.geometry.PointCloud.create_from_depth_image(depth_2, K)

        # depth_1 = o3d.geometry.Image(depth_1)
        # depth_2 = o3d.geometry.Image(depth_2)
        for i in range(gt_match.shape[0]):
            if gt_match[i] == -1:
                continue

            # get depth of plane i
            plane_depth_1 = np.copy(depth_1)
            pixel_index = np.where(plane_seg_1 == i+1)
            z = plane_depth_1[pixel_index]
            x = (pixel_index[1] - 512)*z / 886.81
            y = (pixel_index[0] - 384)*z / 886.81
            xyz = np.zeros((x.shape[0], 3))
            xyz[:, 0] = x
            xyz[:, 1] = y
            xyz[:, 2] = z

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)


            # transform to world coordinate
            pcd = pcd.transform(camera_pose_1)
            # transform to camera 2 coordinate
            pcd = pcd.transform(np.linalg.inv(camera_pose_2))

            # o3d.visualization.draw_geometries([pcd, pcd_2])

            # convert points to numpy
            points_1 = np.asarray(pcd.points)
            # project points to image
            points_1_img = np.matmul(K.intrinsic_matrix, points_1.T).T
            # normalize
            points_1_img = points_1_img / points_1_img[:, 2:]
            points_1_img = points_1_img[:, :2]

            # convert pixel_index to numpy
            # pixel index : (y, x)
            pixel_index = np.asarray(pixel_index).T
            dst_index = np.zeros_like(pixel_index)
            dst_index[:, 0] = points_1_img[:, 0]
            dst_index[:, 1] = points_1_img[:, 1]
            src_index = np.zeros_like(pixel_index)
            src_index[:, 0] = pixel_index[:, 1]
            src_index[:, 1] = pixel_index[:, 0]

            # compute homography
            H, _ = cv2.findHomography(src_index, dst_index, cv2.RANSAC, 1)
            if H is None:
                continue

            H_string = np.array2string(H.reshape([-1]), precision=8, separator=',', suppress_small=True)

            # test
            H_new = np.fromstring(H_string[1:-1], dtype=float, sep=',').reshape([3, 3])
            # save info to dataframe
            df = df.append({'scene_name': scene_name, 'camera_id_1': camera_id_1, 'camera_id_2': camera_id_2,
                            'frame_id_1': frame_id_1, 'frame_id_2': frame_id_2, 'plane_id_1': i+1, 'plane_id_2': int(gt_match[i]+1),
                            'homography': H_string}, ignore_index=True)
            # H = np.linalg.inv(H)
            # pass



            # wrap image using homography

            # H = torch.tensor([[1.0, 0.0, 100], [0.0, 1.0, 100], [0.0, 0.0, 1.0]])

            # im_dst = cv2.warpPerspective(image_1, H, (1024, 768))

            # # show image2 and im_dst using plt
            # plt.subplot(1, 2, 1)
            # plt.title('image_2')
            # plt.imshow(image_2)
            # plt.subplot(1, 2, 2)
            # plt.title('wrapped image_1')
            # plt.imshow(im_dst)
            # plt.show()
            # pass

        pass

    df.to_csv(csv_path, index=True, mode='w', header=True)

if __name__ == "__main__":
    # scene_list = ['ai_001_001', 'ai_001_002', 'ai_001_010']
    root_dir = "/cluster/project/infk/cvg/students/junwang/hypersimLite"
    test_list_path = os.path.join(root_dir, "train_scenes.txt")
    with open(test_list_path, "r") as f:
        test_scene = f.read().splitlines()
    print(test_scene)
    main(test_scene)