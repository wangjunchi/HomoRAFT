import numpy as np
from os import path as osp
import yaml
import torch
import argparse
import h5py
import cv2

from Models.Raft import Model as RAFT
from Utils.metrics import compute_mace, compute_homography

def load_h5(filename):
    '''Loads dictionary from hdf5 file'''
    dict_to_load = {}
    try:
        with h5py.File(filename, 'r') as f:
            keys = [key for key in f.keys()]
            for key in keys:
                dict_to_load[key] = f[key][()]
    except:
        print('Cannot find file {}'.format(filename))
    return dict_to_load

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

    # dataset
    DIR = '/home/junchi/sp1/dataset/'
    dataset = 'EVD'
    split = 'val'
    Hgt = load_h5(f'{DIR}/{dataset}/{split}/Hgt.h5')

    with torch.no_grad():
        for k, H in Hgt.items():
            img1_fname = f'{DIR}/{dataset}/{split}/imgs/1/' + k.split('-')[0] + '.png'
            img2_fname = f'{DIR}/{dataset}/{split}/imgs/2/' + k.split('-')[0] + '.png'
            img1 = cv2.cvtColor(cv2.imread(img1_fname), cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.imread(img2_fname), cv2.COLOR_BGR2RGB)
            # resize image to 320x240
            img1_numpy = cv2.resize(img1, (320, 240))
            img2_numpy = cv2.resize(img2, (320, 240))
            img1 = torch.from_numpy(img1_numpy).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            img2 = torch.from_numpy(img2_numpy).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            flow_pred12, homo_pred, residual, weight = model(img1, img2, None, iters=20)
            final_flow12 = flow_pred12[-1]
            h_pred12 = compute_homography(final_flow12)

            # draw  weight map
            weight = weight[-1]
            weight = weight.squeeze().cpu().numpy()
            weight = cv2.resize(weight, (320, 240))
            weight_map = cv2.applyColorMap((weight * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # convert to rgb
            weight_map = cv2.cvtColor(weight_map, cv2.COLOR_BGR2RGB)

            image_1 = img1_numpy
            image_2 = img2_numpy
            # overlap weight map to image_1
            image_1 = cv2.addWeighted(image_1, 0.7, weight_map, 0.3, 0)
            # cat image_1 and image_2
            image = np.concatenate((image_1, image_2), axis=1)
            pass

            # warp image_1 to image_2 using h_pred12
            # h_pred12 = np.linalg.inv(h_pred12)
            # h_pred12 = h_pred12 / h_pred12[2, 2]
            image_1_warped = cv2.warpPerspective(img1_numpy, h_pred12[0], (320, 240))
            # cat image_1_warped and image_2
            image_warped = np.concatenate((image_1_warped, img2_numpy), axis=1)
            pass


if __name__ == '__main__':
    main()