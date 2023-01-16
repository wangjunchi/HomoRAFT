import numpy as np
from os import path as osp
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import cv2
from tqdm import tqdm
import torch.nn.functional as F

from Dataset.hpatch.hpatch import HPatchesDataset
from Models.Raft import Model as RAFT
from Utils.metrics import compute_mace, compute_homography

def getBestMatches(ref_des, q_des, ratio=0.9):
    bf = cv2.BFMatcher(crossCheck=True)
    # return  bf.match(ref_des, q_des)
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = bf.match(ref_des, q_des)  # first k best matches

    # best_matches = []
    # from Lowe's
    # for i, (m, n) in enumerate(matches):
    #     if m.distance < ratio * n.distance:
    #         best_matches.append(m)
    # # print(type(best_matches[0]))

    return matches

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
    parser.add_argument('--ckpt', type=str, default='../model_29_epoch_no_mask.pth',
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
        res_2x = []
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
            correct_2x = 0
            epes = []
            for _, mini_batch in pbar:
                # get data
                img1 = mini_batch['patch_1']
                img2 = mini_batch['patch_2']
                mask_0 = mini_batch['mask_0']
                mask_1 = mini_batch['mask_1']

                # convert to numpy
                img1 = img1.numpy().transpose(0, 2, 3, 1) * 255
                img1 = img1.astype(np.uint8).squeeze()
                img2 = img2.numpy().transpose(0, 2, 3, 1) * 255
                img2 = img2.astype(np.uint8).squeeze()
                # convert to gray
                gray_1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray_2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                # detect sift keypoints
                # using sift to extract features
                sift = cv2.SIFT_create()
                ref_kp, ref_des = sift.detectAndCompute(gray_1, None)
                q_kp, q_des = sift.detectAndCompute(gray_2, None)
                best_matches = getBestMatches(ref_des, q_des, ratio=0.7)

                img3 = cv2.drawMatches(img1, ref_kp, img2, q_kp, best_matches[:20], None)

                # compute homography
                src_pts = np.float32([ref_kp[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([q_kp[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
                H_hat, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                # print(H)

                four_points = [[0, 0],
                               [img1.shape[1] - 1, 0],
                               [img1.shape[1] - 1, img1.shape[0] - 1],
                               [0, img1.shape[0] - 1]]
                four_points = np.asarray(four_points, dtype=np.float32)

                h_gt = mini_batch['gt_H'].cpu().numpy()
                H = h_gt[0]
                H_inv = np.linalg.inv(H)

                warpped = cv2.perspectiveTransform(np.asarray([four_points]), H_hat).squeeze()
                rewarp = cv2.perspectiveTransform(np.asarray([warpped]), H_inv).squeeze()
                delta = four_points - rewarp
                error = np.linalg.norm(delta, axis=1)
                error = np.mean(error)
                if error <= 3:
                    correct += 1




            print('Scene {} accuracy: {}'.format(id+1, correct/len(test_dataset)))
            res.append(correct / len(test_dataloader))

        print(res)
        print('Average accuracy: {}'.format(np.mean(res)))
        # print('Average accuracy at 2x scale: {}'.format(np.mean(res_2x)))

if __name__ == '__main__':
    main()