import numpy as np
from os import path as osp
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import cv2
from tqdm import tqdm

from Dataset.hpatch.hpatch import HPatchesDataset
from Models.Raft import Model as RAFT
from Utils.metrics import compute_mace, compute_homography



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
    parser.add_argument('--ckpt', type=str, default='../model_24_epoch.pth',
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

    with torch.no_grad():
        number_of_scenes = 5
        res = []
        # loop over scenes (1-2, 1-3, 1-4, 1-5, 1-6)
        for id, k in enumerate(range(2, number_of_scenes + 2)):
            test_dataset = \
                HPatchesDataset(csv_file=osp.join(args.csv_path,
                                                  'hpatches_1_{}.csv'.format(k)),
                                image_path_orig=args.image_data_path,
                                transforms=None,
                                image_size=(240, 320))

            test_dataloader = DataLoader(test_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=0)

            pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
            correct = 0
            for _, mini_batch in pbar:
                # get data
                img1 = mini_batch['patch_1'].to(device)
                img2 = mini_batch['patch_2'].to(device)


                # compute flow
                flow_pred = model(img1, img2, iters=16)
                # compute homography and mace
                final_flow = flow_pred[-1]
                h_pred = compute_homography(final_flow)
                h_gt = mini_batch['gt_H'].cpu().numpy()
                four_points = [[0, 0],
                               [img1.shape[-1]-1, 0],
                               [img1.shape[-1]-1, img1.shape[-2]-1],
                               [0, img1.shape[-2]-1]]
                four_points = np.asarray(four_points, dtype=np.float32)

                for i in range(h_pred.shape[0]):
                    H = h_gt[i]
                    H_inv = np.linalg.inv(H)
                    H_hat = h_pred[i]
                    warpped = cv2.perspectiveTransform(np.asarray([four_points]), H_hat).squeeze()
                    rewarp = cv2.perspectiveTransform(np.asarray([warpped]), H_inv).squeeze()
                    delta = four_points - rewarp
                    error = np.linalg.norm(delta, axis=1)
                    error = np.mean(error)
                    if error < 5:
                        correct += 1
            print('Scene {} accuracy: {}'.format(id+1, correct/len(test_dataset)))
            res.append(correct / len(test_dataloader))



        print(res)
        print('Average accuracy: {}'.format(np.mean(res)))

if __name__ == '__main__':
    main()