import time

import matplotlib.pyplot as plt
import numpy as np
from os import path as osp
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import cv2
from tqdm import tqdm
import torch.nn.functional as F

from Dataset.coco import Coco
from Models.Raft_BA import Model as RAFT_BA
from Models.RAFT import Model as RAFT

from Utils.metrics import compute_mace, compute_homography

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description='Raft evaluation on COCO')

    parser.add_argument('--cfg-file', type=str, default='../configs/raft.yaml',
                        help='path to training transformation csv folder')
    parser.add_argument('--ckpt', type=str, default='../ba_model_29_epoch.pth',
                        help='Checkpoint to use')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='evaluation batch size')
    parser.add_argument('--seed', type=int, default=42,
                        help='default seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    with open(args.cfg_file, 'r') as file:
        config = yaml.full_load(file)
    # Model
    checkpoint_fname = args.ckpt
    if not osp.isfile(checkpoint_fname):
        raise ValueError('check the snapshots path')

    # Import model
    model = RAFT()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(checkpoint_fname, map_location=device)['model'])
    # model = nn.DataParallel(model)
    model.eval()
    model = model.to(device)

    # Dataset
    dataset = Coco(config['data'], 'cpu')
    dataloader = DataLoader(dataset.get_dataset('val'), batch_size=1,
                            shuffle=False, num_workers=8, pin_memory=True)

    correct = 0
    total = 0
    epes = []
    errs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            image_patch_1 = batch['image0']
            image_patch_2 = batch['image1']
            new_homography = batch['homography']

            image_patch_1 = image_patch_1.to(device)
            image_patch_2 = image_patch_2.to(device)

            start = time.time()
            # for i in range(500):
            #     flow_pred12, homo_pred, residual, weights = model(image_patch_1, image_patch_2, None, iters=5)
            # end = time.time()
            # print(end - start)
            # print("FPS: ", 500 / (end - start))
            # print("average time: ", (end - start) / 500)
            # compute homography
            flow_pred12, homo_pred, residual, weights = model(image_patch_1, image_patch_2, None, iters=10)
            final_flow12 = flow_pred12[-1]
            h_pred12 = compute_homography(final_flow12)
            h_pred12_1 = h_pred12

            # flow_pred21, homo_pred, residual, weights = model(image_patch_2, image_patch_1, None, iters=10)
            # final_flow21 = flow_pred21[-1]
            # h_pred21 = compute_homography(final_flow21)
            # h_pred12_2 = np.linalg.inv(h_pred21)

            h_gt = new_homography.numpy()

            # compute accuracy
            H = h_gt[0]
            H_inv = np.linalg.inv(H)
            H_hat = h_pred12_1[0]

            four_points = [[0, 0],
                           [image_patch_1.shape[-1] - 1, 0],
                           [image_patch_1.shape[-1] - 1, image_patch_1.shape[-2] - 1],
                           [0, image_patch_1.shape[-2] - 1]]
            four_points = np.asarray(four_points, dtype=np.float32)

            warpped = cv2.perspectiveTransform(np.asarray([four_points]), H_hat).squeeze()
            rewarp = cv2.perspectiveTransform(np.asarray([warpped]), H_inv).squeeze()
            delta = four_points - rewarp
            error = np.linalg.norm(delta, axis=1)
            error = np.mean(error)
            total += 1
            errs.append(error)
            if error <= 3:
                correct += 1

            # compute average endpoint error
            coords = np.meshgrid(np.arange(image_patch_1.shape[-1]), np.arange(image_patch_1.shape[-2]))  # 240 * 240
            coords = np.stack(coords, axis=-1)
            coords = coords.reshape(-1, 2).astype(np.float32)
            target_flow = cv2.perspectiveTransform(np.asarray([coords]), H).squeeze()
            est_flow = cv2.perspectiveTransform(np.asarray([coords]), H_hat).squeeze()

            # compute mask
            # applying mask
            mask_x_gt = np.logical_and(target_flow[:, 0] >= 0, target_flow[:, 0] <= 320)
            mask_y_gt = np.logical_and(target_flow[:, 1] >= 0, target_flow[:, 1] <= 240)
            mask_gt = mask_x_gt & mask_y_gt
            # mask_gt = np.concatenate((mask_xx_gt[:, None], mask_xx_gt[:, None]), axis=1)
            target_flow = target_flow[mask_gt, :]
            est_flow = est_flow[mask_gt, :]

            epe = np.linalg.norm(target_flow - est_flow, axis=-1)
            epe = np.clip(epe, 0, 320)
            epes.append(np.mean(epe))


        print("Accuracy: ", correct/total)
        print("Average error: ", np.mean(errs))
        print("Median error: ", np.median(errs))
        print("Average epe = ", np.mean(epes))
        print("Median epe = ", np.median(epes))

if __name__ == '__main__':
    main()