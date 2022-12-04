from os import path as osp

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from Dataset.utils.data_reader import resize_and_crop

class HPatchesDataset(Dataset):
    """
    HPatches dataset (for evaluation)
    Args:
        csv_file: csv file with ground-truth data
        image_path_orig: filepath to the dataset (full resolution)
        transforms: image transformations (data preprocessing)
        image_size: size (tuple) of the output images
    Output:
        source_image: source image
        target_image: target image
        correspondence_map: pixel correspondence map
            between source and target views
        mask: valid/invalid correspondences
    """

    def __init__(self,
                 csv_file,
                 image_path_orig,
                 transforms,
                 image_size=(240, 240)):
        self.df = pd.read_csv(csv_file)
        self.image_path_orig = image_path_orig
        self.transforms = transforms
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        obj = str(data.obj)
        im1_id, im2_id = str(data.im1), str(data.im2)

        H = data[5:].astype('double').values.reshape((3, 3))

        # read the images
        img1 = cv2.imread(osp.join(self.image_path_orig,
                                           obj,
                                           im1_id + '.ppm'), -1)
        img2 = cv2.imread(osp.join(self.image_path_orig,
                                   obj,
                                   im2_id + '.ppm'), -1)

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # transform the homography
        H = self.adapt_homography_to_preprocessing(H, img1.shape[:2],
                                                   img2.shape[:2])

        # resize and crop
        img1 = resize_and_crop(img1, self.image_size)
        img2 = resize_and_crop(img2, self.image_size)

        img1 = img1.astype('float32') / 255.
        img2 = img2.astype('float32') / 255.

        # global transforms
        if self.transforms is not None:
            img1 = self.transforms(img1)
            img2 = self.transforms(img2)
        else:
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        return {'patch_1': img1,
                'patch_2': img2,
                'gt_H': H
                }

    def adapt_homography_to_preprocessing(self, H, img_shape0, img_shape1):
        source_size0 = np.array(img_shape0, dtype=float)
        source_size1 = np.array(img_shape1, dtype=float)
        target_size = np.array(self.image_size, dtype=float)

        # Get the scaling factor in resize
        scale0 = np.amax(target_size / source_size0)
        scaling0 = np.diag([1. / scale0, 1. / scale0, 1.]).astype(float)
        scale1 = np.amax(target_size / source_size1)
        scaling1 = np.diag([scale1, scale1, 1.]).astype(float)

        # Get the translation params in crop
        pad_y0 = (source_size0[0] * scale0 - target_size[0]) / 2.
        pad_x0 = (source_size0[1] * scale0 - target_size[1]) / 2.
        translation0 = np.array([[1., 0., pad_x0],
                                 [0., 1., pad_y0],
                                 [0., 0., 1.]], dtype=float)
        pad_y1 = (source_size1[0] * scale1 - target_size[0]) / 2.
        pad_x1 = (source_size1[1] * scale1 - target_size[1]) / 2.
        translation1 = np.array([[1., 0., -pad_x1],
                                 [0., 1., -pad_y1],
                                 [0., 0., 1.]], dtype=float)

        return translation1 @ scaling1 @ H @ scaling0 @ translation0

if __name__ == '__main__':
    # dataset
    dataset = HPatchesDataset(csv_file='csv/hpatches_1_2.csv',
                      image_path_orig='/home/junchi/sp1/dataset/hpatches-sequences-release',
                      transforms=None,
                      image_size=(240, 320))

    # dataloader
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)

    # iterate over the dataset
    for i_batch, sample_batched in enumerate(dataloader):
        source_image = sample_batched['patch_1']
        target_image = sample_batched['patch_2']


        print(source_image.shape)
        print(target_image.shape)

        img1 = source_image.numpy()[0].transpose(1, 2, 0)
        img1 = (img1 * 255).astype(np.uint8)
        img2 = target_image.numpy()[0].transpose(1, 2, 0)
        img2 = (img2 * 255).astype(np.uint8)

        H = sample_batched['gt_H'].numpy()[0]
        # apply homography
        img1_warped = cv2.warpPerspective(img1, H, (img1.shape[1], img1.shape[0]))


        # visualize
        plt.figure()
        plt.imshow(img1)
        plt.figure()
        plt.imshow(img2)
        plt.figure()
        plt.imshow(img1_warped)
        plt.show()

        if i_batch == 0:
            break

    print('Done!')