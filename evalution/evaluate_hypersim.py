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

from Dataset.hypersim.hypersim_planes import HypersimPlaneDataset
from Models.Raft import Model as RAFT
from Utils.metrics import compute_mace, compute_homography

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description='Raft evaluation on HPatches')
    # Paths
    parser.add_argument('--csv-path', type=str, default='../Dataset/hpatch/csv',
                        help='path to training transformation csv folder')
    parser.add_argument('--cfg-file', type=str, default='../configs/raft.yaml',
                        help='path to training transformation csv folder')
    parser.add_argument('--image-data-path', type=str,
                        default='/home/junchi/sp1/dataset/hpatches-sequences-release',
                        help='path to folder containing training images')
    parser.add_argument('--ckpt', type=str, default='../hypersim_model_24_epoch.pth',
                        help='Checkpoint to use')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='evaluation batch size')
    parser.add_argument('--seed', type=int, default=1984, help='Pseudo-RNG seed')

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

    scene_list = ['ai_001_010']
    dataset = HypersimPlaneDataset("/home/junchi/sp1/dataset/hypersim", scene_list)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    correct = 0
    total = 0
    epes = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            image_patch_1 = batch['image0']
            image_patch_2 = batch['image1']
            new_homography = batch['homography']
            mask_1 = batch['seg0']
            mask_2 = batch['seg1']

            image_patch_1 = image_patch_1 * mask_1[:, None, :, :]
            image_patch_2 = image_patch_2 * mask_2[:, None, :, :]
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

            flow_pred21, homo_pred, residual, weights = model(image_patch_2, image_patch_1, None, iters=10)
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
            total += 1
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



            # draw weight
            weight = weights[-1]
            weight = weight[0].squeeze().cpu().numpy()
            weight = cv2.resize(weight, (320, 240))
            weight_map = cv2.applyColorMap((weight * 255).astype(np.uint8), cv2.COLORMAP_JET)
            weight_map = cv2.cvtColor(weight_map, cv2.COLOR_BGR2RGB)
            image_1 = image_patch_1.squeeze().cpu().numpy()
            image_1 = np.transpose(image_1, (1, 2, 0))
            image_1 = (image_1 * 255).astype(np.uint8)
            image_2 = image_patch_2.squeeze().cpu().numpy()
            image_2 = np.transpose(image_2, (1, 2, 0))
            image_2 = (image_2 * 255).astype(np.uint8)
            # overlap weight map to image_1
            image_1 = cv2.addWeighted(image_1, 0.7, weight_map, 0.3, 0)
            # cat image_1 and image_2
            image = np.concatenate((image_1, image_2), axis=1)
            # cv2.imshow("image", image)

            image_1_numpy = batch['image0'].permute(0, 2, 3, 1).squeeze().numpy()
            image_1_numpy = (image_1_numpy * 255).astype(np.uint8)
            image_2_numpy = batch['image1'].permute(0, 2, 3, 1).squeeze().numpy()
            image_2_numpy = (image_2_numpy * 255).astype(np.uint8)
            mask_1_numpy = batch['seg0'].cpu().numpy()[0]
            mask_2_numpy = batch['seg1'].cpu().numpy()[0]
            final_flow12 = flow_pred12[-1]
            image_1_warped = cv2.warpPerspective(image_1_numpy, H_hat, (320, 240))
            mask_1_warped = cv2.warpPerspective(mask_1_numpy, H_hat, (320, 240))
            # visualize with plt
            plt.subplot(2, 3, 1)
            plt.title("image_1")
            plt.imshow(image_1_numpy/255.0)
            plt.subplot(2, 3, 2)
            plt.title("image_2")
            plt.imshow(image_2_numpy/255.0)
            plt.subplot(2, 3, 3)
            plt.title("image_1_warped")
            plt.imshow(image_1_warped/255.0)
            plt.subplot(2, 3, 4)
            plt.title("mask_1")
            plt.imshow(mask_1_numpy)
            plt.subplot(2, 3, 5)
            plt.title("mask_2")
            plt.imshow(mask_2_numpy)
            plt.subplot(2, 3, 6)
            plt.title("mask_1_warped")
            plt.imshow(mask_1_warped)
            plt.show()
            pass

        print("Accuracy: ", correct/total)
        print("Average epe = ", np.mean(epes))

if __name__ == '__main__':
    main()