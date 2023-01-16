import numpy as np
from os import path as osp
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import cv2
from tqdm import tqdm
import torch.nn.functional as F

# add root path to sys.path
import sys
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

from Dataset.hpatch.hpatch import HPatchesDataset
from Models.Raft import Model as RAFT
from Utils.metrics import compute_mace, compute_homography



def main(iterations = 10):

    # Argument parsing
    parser = argparse.ArgumentParser(description='Raft evaluation on HPatches')
    # Paths
    parser.add_argument('--csv-path', type=str, default='./Dataset/hpatch/csv',
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
        res_thres =[[] for i in range(10)]
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
            corret_thres = [0 for i in range(10)]
            iteration = iterations
            for _, mini_batch in pbar:
                # get data
                img1 = mini_batch['patch_1'].to(device)
                img2 = mini_batch['patch_2'].to(device)
                mask_0 = mini_batch['mask_0'].to(device)
                mask_1 = mini_batch['mask_1'].to(device)


                # compute flow
                flow_pred12, homo_pred, residual = model(img1, img2, mask_0, iters=iteration)

                # flow_pred12, homo_pred, residual, weight = model(img1, img2, mask_0, iters=iteration)
                final_flow12 = flow_pred12[-1]

                h_pred12 = compute_homography(final_flow12)
                h_pred12_1 = h_pred12

                # weight = weight[-1]
                # # draw weight map
                # weight = weight.squeeze().cpu().numpy()
                # weight = cv2.resize(weight, (320, 240))
                # weight_map = cv2.applyColorMap((weight * 255).astype(np.uint8), cv2.COLORMAP_JET)
                # # convert to rgb
                # weight_map = cv2.cvtColor(weight_map, cv2.COLOR_BGR2RGB)
                # # prepare image
                # image_1 = mini_batch['patch_1'].squeeze().numpy()
                # image_1 = np.transpose(image_1, (1, 2, 0))
                # image_1 = (image_1 * 255).astype(np.uint8)
                # image_2 = mini_batch['patch_2'].squeeze().numpy()
                # image_2 = np.transpose(image_2, (1, 2, 0))
                # image_2 = (image_2 * 255).astype(np.uint8)
                # # overlap weight map to image_1
                # image_1 = cv2.addWeighted(image_1, 0.7, weight_map, 0.3, 0)
                # # cat image_1 and image_2
                # image = np.concatenate((image_1, image_2), axis=1)

                flow_pred21, homo_pred, residual = model(img2, img1, mask_1, iters=iteration)
                # flow_pred21, homo_pred, residual, weight = model(img2, img1, mask_1, iters=iteration)
                final_flow21 = flow_pred21[-1]
                h_pred21 = compute_homography(final_flow21)
                h_pred12_2 = np.linalg.inv(h_pred21)
                h_gt = mini_batch['gt_H'].cpu().numpy()
                four_points = [[0, 0],
                               [img1.shape[-1]-1, 0],
                               [img1.shape[-1]-1, img1.shape[-2]-1],
                               [0, img1.shape[-2]-1]]
                four_points = np.asarray(four_points, dtype=np.float32)


                for i in range(h_pred12_1.shape[0]):
                    H = h_gt[i]
                    H_inv = np.linalg.inv(H)
                    H_hat = (h_pred12_1[i] + h_pred12_2[i])/2
                    # H_hat = h_pred12_1[i]

                    warpped = cv2.perspectiveTransform(np.asarray([four_points]), H_hat).squeeze()
                    rewarp = cv2.perspectiveTransform(np.asarray([warpped]), H_inv).squeeze()
                    delta = four_points - rewarp
                    error = np.linalg.norm(delta, axis=1)
                    error = np.mean(error)
                    if error <= 3:
                        correct += 1

                    for i in range(10):
                        if error <= i+1:
                            corret_thres[i] += 1



                    # compute the average endpoint error
                    S = np.array([0.75, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)
                    # scale the ground truth homography
                    H = np.matmul(S, np.matmul(H, np.linalg.inv(S)))
                    H_hat = np.matmul(S, np.matmul(H_hat, np.linalg.inv(S)))
                    coords = np.meshgrid(np.arange(img1.shape[-2]), np.arange(img1.shape[-2])) # 240 * 240
                    coords = np.stack(coords, axis=-1)
                    coords = coords.reshape(-1, 2).astype(np.float32)
                    target_flow = cv2.perspectiveTransform(np.asarray([coords]), H).squeeze()
                    est_flow = cv2.perspectiveTransform(np.asarray([coords]), H_hat).squeeze()

                    # compute mask
                    # applying mask
                    mask_x_gt = np.logical_and(target_flow[:, 0] >= 0, target_flow[:, 0] <= 240)
                    mask_y_gt = np.logical_and(target_flow[:, 1] >= 0, target_flow[:, 1] <= 240)
                    mask_gt = mask_x_gt & mask_y_gt
                    # mask_gt = np.concatenate((mask_xx_gt[:, None], mask_xx_gt[:, None]), axis=1)
                    target_flow = target_flow[mask_gt, :]
                    est_flow = est_flow[mask_gt, :]

                    epe = np.linalg.norm(target_flow - est_flow, axis=-1)
                    epe = np.clip(epe, 0, 240)
                    # if np.mean(epe) < 100:
                    epes.append(np.mean(epe))
            print('Scene {} accuracy: {}'.format(id+1, correct/len(test_dataset)))
            # print('Scene {} accuracy at 2x scale: {}'.format(id+1, correct_2x/len(test_dataset)))
            print('Scene {} average epe: {}'.format(id+1, np.mean(epes)))
            res.append(correct / len(test_dataloader))
            res_2x.append(correct_2x / len(test_dataloader))

            for i in range(10):
                res_thres[i].append(corret_thres[i] / len(test_dataloader))



        print(res)
        print('Average accuracy: {}'.format(np.mean(res)))
        # print('Average accuracy at 2x scale: {}'.format(np.mean(res_2x)))

        for i in range(10):
            print('Average accuracy at {} pixel: {}'.format(i+1, np.mean(res_thres[i])))

if __name__ == '__main__':
    # test_iterations = [1,3,5,8,10,12,14,16,20]
    test_iterations = [10]
    for i in test_iterations:
        print('Test iteration {}'.format(i))
        main(i)
        print('---------------------')
