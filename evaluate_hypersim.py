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
import os
import nonechucks as nc


from Dataset.hypersim.hypersim_planes import HypersimPlaneDataset
from Models.Raft import Model as RAFT
from Utils.metrics import compute_mace, compute_homography

def collate_fn(batch):
    if len(batch)!=0:
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
    else:
        return None


def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description='Raft evaluation on HPatches')
    # Paths
    parser.add_argument('--csv-path', type=str, default='../Dataset/hpatch/csv',
                        help='path to training transformation csv folder')
    parser.add_argument('--cfg-file', type=str, default='../configs/raft.yaml',
                        help='path to training transformation csv folder')
    parser.add_argument('--ckpt', type=str, default='/cluster/project/infk/cvg/students/junwang/HomoRAFT/log/hypersim-raft-lr-1e-4-22-12-23-09_27_02/model_24_epoch.pth',
                        help='Checkpoint to use')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='evaluation batch size')
    parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

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

    dataset_dir = '/cluster/project/infk/cvg/students/junwang/hypersim'
    test_list_path = os.path.join(dataset_dir, "test_scenes.txt")
    with open(test_list_path, "r") as f:
        scene_list = f.read().splitlines()
    print(scene_list)
    dataset = HypersimPlaneDataset(dataset_dir, scene_list)
    dataset = nc.SafeDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    # dataloader = nc.SafeDataLoader(dataset, batch_size=1, shuffle=False)
    correct = 0
    total = 0
    epes = []
    errs = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=True):
            # if idx == 200:
            #     break
            if idx % 100 == 1:
                print("Acc = ", correct/total)
            if batch is None:
                continue
            image_patch_1 = batch['image0']
            image_patch_2 = batch['image1']
            new_homography = batch['homography']
            mask_1 = batch['seg0']
            mask_2 = batch['seg1']

            image_patch_1 = image_patch_1 * mask_1[:, None, :, :]
            image_patch_2 = image_patch_2 * mask_2[:, None, :, :]
            image_patch_1 = image_patch_1.to(device)
            image_patch_2 = image_patch_2.to(device)
            flow_pred12, homo_pred, residual = model(image_patch_1, image_patch_2, None, iters=10)
            # compute homography
            final_flow12 = flow_pred12[-1]
            h_pred12 = compute_homography(final_flow12)
            h_pred12_1 = h_pred12

            flow_pred21, homo_pred, residual = model(image_patch_2, image_patch_1, None, iters=10)
            final_flow21 = flow_pred21[-1]
            h_pred21 = compute_homography(final_flow21)
            h_pred12_2 = np.linalg.inv(h_pred21)

            h_gt = new_homography.numpy()

            # compute accuracy
            H = h_gt[0]
            H_inv = np.linalg.inv(H)
            H_hat = (h_pred12_1[0] + h_pred12_2[0])/2

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
            errs.append(error)
            # print("mean err = ", error)
            total += 1
            if error <= 3:
                correct += 1
            # if error < 1000:
            #     errs.append(error)

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
    print("Average reprojection error = ", np.mean(errs))
    print("Median reprojection error = ", np.median(errs))
    print("Accuracy: ", correct/total)
    print("Average epe = ", np.mean(epes))
    print("Median epe = ", np.median(epes))

if __name__ == '__main__':
    main()