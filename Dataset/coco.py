""" MS COCO dataset. """

import os
import numpy as np
import torch
import cv2
from pathlib import Path
from torch.utils.data import Dataset

from Dataset.base_dataset import BaseDataset
from Dataset.utils.homographies import sample_homography, compute_valid_mask
from Dataset.utils.data_reader import resize_and_crop
from Dataset.utils.data_augmentation import photometric_augmentation
from Dataset.utils.visualization import draw_matching


class Coco(BaseDataset):
    def __init__(self, config, device):
        super().__init__(config, device)
        root_dir = Path(os.path.expanduser(config['data_path']))
        self._paths = {}

        # Train split
        train_dir = Path(root_dir, 'train2014')
        self._paths['train'] = [str(p) for p in list(train_dir.iterdir())]

        # Val split
        val_dir = Path(root_dir, 'val2014')
        val_images = list(val_dir.iterdir())
        self._paths['val'] = [str(p)
                              for p in val_images[:config['sizes']['val']]]

        # Test split
        self._paths['test'] = [str(p)
                               for p in val_images[-config['sizes']['test']:]]

    def get_dataset(self, split):
        return _Dataset(self._paths[split], self._config, self._device)


class _Dataset(Dataset):

    def __init__(self, paths, config, device):
        self._paths = paths
        self._config = config
        self._angle_lim = np.pi / 4
        self._device = device

    def __getitem__(self, item):
        img0 = cv2.imread(self._paths[item])
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

        # Resize the image
        if 'img_size' in self._config:
            img0 = resize_and_crop(img0, self._config['img_size'])
        img_size = img0.shape[:2]

        H, rot_angle = sample_homography(
            img_size, **self._config['warped_pair']['params'])

        # check if the homography is valid
        H_inv = np.linalg.inv(H)
        valid_mask = compute_valid_mask(H_inv, img_size)
        while valid_mask.sum() < 0.1 * img_size[0] * img_size[1]:
            # print('Invalid homography, overlap ratio: {} ,resampling...'.format(valid_mask.sum() / (img_size[0] * img_size[1])))
            H, rot_angle = sample_homography(
                img_size, **self._config['warped_pair']['params'])
            H_inv = np.linalg.inv(H)
            rot_angle = np.clip(np.abs(rot_angle) / self._angle_lim, 0., 1.)
            valid_mask = compute_valid_mask(H_inv, img_size)
        # print('Valid homography, overlap ratio: {:.2f}'.format(valid_mask.sum() / (img_size[0] * img_size[1])))


        rot_angle = np.clip(np.abs(rot_angle) / self._angle_lim, 0., 1.)
        self._config['warped_pair']['params']['rotation'] = True
        H_inv = np.linalg.inv(H)
        img1 = cv2.warpPerspective(img0, H, (img_size[1], img_size[0]),
                                   flags=cv2.INTER_LINEAR)

        # Apply photometric augmentation
        config_aug = self._config['photometric_augmentation']
        if config_aug['enable']:
            img0 = photometric_augmentation(img0, config_aug)
            img1 = photometric_augmentation(img1, config_aug)

        # Normalize
        img0 = img0.astype(float) / 255.
        img1 = img1.astype(float) / 255.

        # Generate matching and flow
        matching, flow = self.generate_matching_and_flow(H)

        outputs = {}

        # Compute valid masks

        valid_mask0 = compute_valid_mask(
            H_inv, img_size,
            self._config['warped_pair']['valid_border_margin'])
        outputs['valid_mask0'] = torch.tensor(
            valid_mask0, dtype=torch.float, device=self._device)
        valid_mask1 = compute_valid_mask(
            H, img_size,
            self._config['warped_pair']['valid_border_margin'])
        outputs['valid_mask1'] = torch.tensor(
            valid_mask1, dtype=torch.float, device=self._device)

        outputs['image0'] = torch.tensor(
            img0.transpose(2, 0, 1), dtype=torch.float,
            device=self._device)
        outputs['image1'] = torch.tensor(
            img1.transpose(2, 0, 1), dtype=torch.float,
            device=self._device)
        outputs['homography'] = torch.tensor(H, dtype=torch.float,
                                             device=self._device)

        # Useful additional information
        outputs['rot_angle'] = torch.tensor([rot_angle], dtype=torch.float,
                                            device=self._device)

        outputs['matching'] = torch.tensor(matching, dtype=torch.float,
                                          device=self._device)
        outputs['flow'] = torch.tensor(flow, dtype=torch.float,
                                       device=self._device).permute(2, 0, 1)

        return outputs

    def __len__(self):
        return len(self._paths)

    def generate_matching_and_flow(self, H):
        # coordinate defined as (x, y)
        origin_coords = np.meshgrid(np.arange(self._config['img_size'][1]), np.arange(self._config['img_size'][0]))
        origin_coords = np.stack(origin_coords, axis=2).reshape(-1, 2)
        origin_coords = np.concatenate([origin_coords, np.ones((origin_coords.shape[0], 1))], axis=1)

        wrapped_coords = np.dot(H, origin_coords.T).T

        origin_coords = origin_coords[:, :2]
        wrapped_coords = wrapped_coords[:, :2] / wrapped_coords[:, 2:]

        origin_coords = origin_coords.reshape(self._config['img_size'][0], self._config['img_size'][1], 2)
        wrapped_coords = wrapped_coords.reshape(self._config['img_size'][0], self._config['img_size'][1], 2)

        flow = wrapped_coords - origin_coords
        matching = wrapped_coords
        return matching, flow



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import yaml
    import matplotlib.pyplot as plt

    config = yaml.load(open('dataset_config.yaml'), Loader=yaml.FullLoader)
    dataset = Coco(config['data'], 'cpu')
    dataloader = DataLoader(dataset.get_dataset('train'),
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)
    print("total number of images: ", len(dataset.get_dataset('train')))
    for step, batch in enumerate(dataloader):
        print(batch['image0'].shape)
        print(batch['image1'].shape)
        print(batch['homography'].shape)
        print(batch['valid_mask0'].shape)
        print(batch['valid_mask1'].shape)
        print(batch['rot_angle'])
        plt.imshow(batch['image0'][0].permute(1, 2, 0).numpy())
        plt.show()
        plt.imshow(batch['image1'][0].permute(1, 2, 0).numpy())
        plt.show()
        plt.imshow(batch['valid_mask0'][0].numpy())
        plt.show()
        plt.imshow(batch['valid_mask1'][0].numpy())
        plt.show()

        # draw matching
        plt.figure(dpi=300)
        image0 = batch['image0'][0].permute(1, 2, 0).numpy()
        image1 = batch['image1'][0].permute(1, 2, 0).numpy()
        matching = batch['matching'][0].numpy()
        mask = batch['valid_mask0'][0].numpy()
        img_match = draw_matching(image0, image1, matching, mask)
        plt.imshow(img_match)
        plt.show()
        pass

