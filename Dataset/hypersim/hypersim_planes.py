import cv2
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import h5py
import torch
import pandas as pd
import itertools
import random
import matplotlib.pyplot as plt
import ast


class HypersimPlaneDataset(Dataset):
    def __init__(self, data_dir, scene_list, image_size=(320, 240)):
        self.data_dir = data_dir
        self.scenes = scene_list
        self.samples = self.get_sample_list()
        self.image_size = image_size

        # self.image_path = "{}/{}/{}/original.png"
        # self.seg_path = "{}/{}/{}/plane_seg.npy"

        self.image_path = "{}/images/scene_{}_final_preview/frame.{}.tonemap.jpg"
        self.seg_path = "{}/images/scene_{}_geometry_hdf5/frame.{}.planes_hires.hdf5"

    # def get_dirs(self):
    #     dirs = []
    #     for root, dirs, files in os.walk(self.data_dir):
    #         print("Found dir: ", dirs)
    #         break
    #     return dirs

    def get_sample_list(self):
        scenes = []
        for scene in self.scenes:
            scene_path = os.path.join(self.data_dir, scene)
            scene_list = pd.read_csv(os.path.join(scene_path, "homography.csv"))
            scenes.append(scene_list)
        sample_list = pd.concat(scenes)
        return sample_list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        scene_name = row["scene_name"]

        camera_id_1 = row["camera_id_1"]
        frame_id_1 = row["frame_id_1"]
        plane_id_1 = row["plane_id_1"]

        camera_id_2 = row["camera_id_2"]
        frame_id_2 = row["frame_id_2"]
        plane_id_2 = row["plane_id_2"]

        homography = row["homography"]
        homography = np.fromstring(homography[1:-1], dtype=float, sep=',').reshape([3, 3])

        image_1 = self.load_image(scene_name, camera_id_1, frame_id_1)
        segment_1 = self.load_segment(scene_name, camera_id_1, frame_id_1, plane_id_1)
        # image_1 *= segment_1[:, :, None]

        segment_2 = self.load_segment(scene_name, camera_id_2, frame_id_2, plane_id_2)
        image_2 = self.load_image(scene_name, camera_id_2, frame_id_2)

        # resize image and segment to 192*256
        image_1 = cv2.resize(image_1, (512, 384))
        image_2 = cv2.resize(image_2, (512, 384))
        segment_1 = cv2.resize(segment_1, (512, 384), interpolation=cv2.INTER_NEAREST)
        segment_2 = cv2.resize(segment_2, (512, 384), interpolation=cv2.INTER_NEAREST)

        s1 = np.array([[1 / 2.0, 0, 0], [0, 1 / 2.0, 0], [0, 0, 1]])
        s2 = np.array([[1 / 2.0, 0, 0], [0, 1 / 2.0, 0], [0, 0, 1]])
        homography = np.matmul(s2, np.matmul(homography, np.linalg.inv(s1)))

        # create bounding box of size 320*240 from mask
        mask = segment_1.astype('bool')
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        center = np.array([(y0 + y1) / 2, (x0 + x1) / 2]).astype('int')
        origin_x_1 = max(0, center[1] - 160)
        origin_y_1 = max(0, center[0] - 120)
        image_1_patch = image_1[origin_y_1:origin_y_1 + 240, origin_x_1:origin_x_1 + 320, :]
        # padding to 128*128
        if image_1_patch.shape[0] < 240:
            image_1_patch = np.pad(image_1_patch, ((0, 240 - image_1_patch.shape[0]), (0, 0), (0, 0)), 'constant')
        if image_1_patch.shape[1] < 320:
            image_1_patch = np.pad(image_1_patch, ((0, 0), (0, 320 - image_1_patch.shape[1]), (0, 0)), 'constant')
        segment_1 = segment_1[origin_y_1:origin_y_1 + 240, origin_x_1:origin_x_1 + 320]
        # padding to 240*320
        if segment_1.shape[0] < 240:
            segment_1 = np.pad(segment_1, ((0, 240 - segment_1.shape[0]), (0, 0)), 'constant')
        if segment_1.shape[1] < 320:
            segment_1 = np.pad(segment_1, ((0, 0), (0, 320 - segment_1.shape[1])), 'constant')

        mask = segment_2.astype('bool')
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        center = np.array([(y0 + y1) / 2, (x0 + x1) / 2]).astype('int')
        origin_x_2 = max(0, center[1] - 160)
        origin_y_2 = max(0, center[0] - 120)
        image_2_patch = image_2[origin_y_2:origin_y_2 + 240, origin_x_2:origin_x_2 + 320, :]
        # padding to 240*320
        if image_2_patch.shape[0] < 240:
            image_2_patch = np.pad(image_2_patch, ((0, 240 - image_2_patch.shape[0]), (0, 0), (0, 0)), 'constant')
        if image_2_patch.shape[1] < 320:
            image_2_patch = np.pad(image_2_patch, ((0, 0), (0, 320 - image_2_patch.shape[1]), (0, 0)), 'constant')
        segment_2 = segment_2[origin_y_2:origin_y_2 + 240, origin_x_2:origin_x_2 + 320]
        # padding to 240*320
        if segment_2.shape[0] < 240:
            segment_2 = np.pad(segment_2, ((0, 240 - segment_2.shape[0]), (0, 0)), 'constant')
        if segment_2.shape[1] < 320:
            segment_2 = np.pad(segment_2, ((0, 0), (0, 320 - segment_2.shape[1])), 'constant')

        t2 = np.array([[1, 0, -origin_x_2], [0, 1, -origin_y_2], [0, 0, 1]])
        t1 = np.array([[1, 0, -origin_x_1], [0, 1, -origin_y_1], [0, 0, 1]])
        homography_new = np.matmul(t2, np.matmul(homography, np.linalg.inv(t1)))

        # use Dilation to enlarge the mask
        kernel = np.ones((10, 10), np.uint8)
        segment_1 = cv2.dilate(segment_1, kernel, iterations=3)
        segment_2 = cv2.dilate(segment_2, kernel, iterations=3)

        # generate matching
        gt_flow, mask = self.generate_gt_match(image_1_patch, homography_new)

        # # convert to tensor
        # image_1_patch = torch.from_numpy(image_1_patch).permute(2, 0, 1).float()
        # image_2_patch = torch.from_numpy(image_2_patch).permute(2, 0, 1).float()

        # check if the sample is valid
        # check homography is valid
        if (np.linalg.matrix_rank(homography_new) != 3):
            return None
        # check overlap is enough
        if (torch.sum(mask) < 0.1 * mask.shape[0] * mask.shape[1]):
            return None

        # convert to tensor
        image_1_patch = torch.from_numpy(image_1_patch).permute(2, 0, 1).float()
        image_2_patch = torch.from_numpy(image_2_patch).permute(2, 0, 1).float()
        image_1_patch = image_1_patch / 255.0
        image_2_patch = image_2_patch / 255.0
        segment_1 = torch.from_numpy(segment_1).float()
        segment_2 = torch.from_numpy(segment_2).float()
        gt_flow = gt_flow.permute(2, 0, 1).float()
        mask = mask.float()
        homography_new = torch.from_numpy(homography_new).float()


        return {'image0': image_1_patch,
                'image1': image_2_patch,
                'flow': gt_flow,
                'homography': homography_new,
                'seg0': segment_1,
                'seg1': segment_2,
                'valid_mask0': mask
                }



    def load_image(self, scene_name, camera_id, frame_id):
        frame_id = str(frame_id).zfill(4)
        img_path = os.path.join(self.data_dir, self.image_path.format(scene_name, camera_id, frame_id))
        assert os.path.exists(img_path), "Image file does not exist: {}".format(img_path)
        image_data = cv2.imread(img_path).astype('float32')
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        return image_data

    # def load_segment(self, scene_name, camera_id, frame_id, plane_id):
    #     frame_id = str(frame_id).zfill(4)
    #     seg_path = os.path.join(self.data_dir, self.seg_path.format(scene_name, camera_id, frame_id))
    #     assert os.path.exists(seg_path), "Segment file does not exist: {}".format(seg_path)
    #     seg_data = np.load(seg_path)
    #     seg_data[seg_data != plane_id] = 0
    #     seg_data[seg_data == plane_id] = 1
    #     return seg_data

    def load_segment(self, scene_name, camera_id, frame_id, plane_id):
        frame_id = str(frame_id).zfill(4)
        seg_path = os.path.join(self.data_dir, self.seg_path.format(scene_name, camera_id, frame_id))
        assert os.path.exists(seg_path), "Segment file does not exist: {}".format(seg_path)
        planes_data = h5py.File(seg_path, 'r')
        planes_data = np.array(planes_data["dataset"])
        seg_data = planes_data[:, :, 1].astype('uint8')
        seg_data[seg_data != plane_id] = 0
        seg_data[seg_data == plane_id] = 1
        return seg_data

    def generate_gt_match(self, patch_1, H):
        h, w, _ = patch_1.shape

        # estimate the grid
        X, Y = np.meshgrid(np.linspace(0, w - 1, w),
                           np.linspace(0, h - 1, h))
        X, Y = X.flatten(), Y.flatten()

        # create matrix representation
        XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T

        # multiply Hinv to XYhom to find the warped grid
        XYwarpHom = np.matmul(H, XYhom)

        # vector representation
        XwarpHom = torch.from_numpy(XYwarpHom[0, :]).float()
        YwarpHom = torch.from_numpy(XYwarpHom[1, :]).float()
        ZwarpHom = torch.from_numpy(XYwarpHom[2, :]).float()

        # Xwarp = \
        #     (2 * XwarpHom / (ZwarpHom + 1e-8) / (w_scale - 1) - 1)
        # Ywarp = \
        #     (2 * YwarpHom / (ZwarpHom + 1e-8) / (h_scale - 1) - 1)
        Xwarp = (XwarpHom / (ZwarpHom + 1e-8))
        Ywarp = (YwarpHom / (ZwarpHom + 1e-8))
        # and now the grid
        gt_match = torch.stack([Xwarp.view(h, w),
                               Ywarp.view(h, w)], dim=-1)

        # mask
        # mask = gt_match.ge(0) & gt_match.le(w)
        mask_x = gt_match[:, :, 0].ge(0) & gt_match[:, :, 0].le(w)
        mask_y = gt_match[:, :, 1].ge(0) & gt_match[:, :, 1].le(h)
        mask = mask_y & mask_x

        X_flow = (XwarpHom / (ZwarpHom + 1e-8)) - X
        Y_flow = (YwarpHom / (ZwarpHom + 1e-8)) - Y
        gt_flow = torch.stack([X_flow.view(h, w),
                                Y_flow.view(h, w)], dim=-1)

        return gt_flow, mask