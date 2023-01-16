import argparse
import yaml
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime

from Dataset.hypersim.hypersim_planes import HypersimPlaneDataset
from Models.Raft import Model as Raft

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from Utils.loss import sequence_loss, residual_loss
from Utils.checkpoint import CheckPointer
from Utils.metrics import compute_mace, compute_homography
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# from torch.multiprocessing import Pool, Process, set_start_method
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def train_one_epoch(model, train_dataloader, optimizer, scheduler, loss_fn, cur_epoch, steps_per_epoch, summary_writer):
    model.train()

    epoch_loss = []
    for iter_no, batch in tqdm(enumerate(train_dataloader), total=steps_per_epoch, disable=True):
        # global step
        step = cur_epoch * steps_per_epoch + iter_no + 1

        image_0 = batch['image0']
        seg_0 = batch['seg0']
        image_1 = batch['image1']
        seg_1 = batch['seg1']
        # mask out non-planar regions
        image_0 = image_0 * seg_0[:, None, :, :]
        image_1 = image_1 * seg_1[:, None, :, :]

        mask_0 = batch['valid_mask0']
        # mask_0 = mask_0 * seg_0
        flow_gt = batch['flow']

        B, _, H, W = image_0.shape
        # model forward
        image_0 = image_0.cuda()
        image_1 = image_1.cuda()
        mask_0 = mask_0.cuda()
        flow_pred, homo_pred, residual = model(image_0, image_1, None, iters=5)
        # loss
        flow_gt = flow_gt.cuda()
        
        loss = 0.0
        loss_flow = 0.0  # only for output
        if loss_fn == 'sequence_loss':
            loss += sequence_loss(flow_pred, flow_gt, mask=mask_0, gamma=0.9)
        elif loss_fn == 'combined_loss':
            loss = sequence_loss(flow_pred, flow_gt, mask=mask_0, gamma=0.8)
            loss_flow = loss.item()
            # # compute corner_loss
            # four_points = torch.tensor([[0, 0], [W//8-1, 0], [W//8-1, H//8-1], [0, H//8-1]], dtype=torch.float32).cuda()
            # four_points = four_points.unsqueeze(0).repeat(B, 1, 1)
            # gt_h = batch['homography'].cuda()
            # loss += corner_loss(homo_pred, gt_h, four_points, gamma=0.9) * 0.1
            # compute residual loss
            loss += residual_loss(residual, mask=mask_0, gamma=0.8) * 5
        else:
            assert False, 'loss_fn not supported'

        if torch.isnan(loss) or loss > 100:
            # skip nan loss
            print("skip large or nan loss")
            continue

        epoch_loss.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()
        # gradient clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), 150)
        optimizer.step()
        # log
        if step % 50 == 0:
            # Calc norm of gradients
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)

            # compute homography and mace
            final_flow = flow_pred[-1]
            h_pred = compute_homography(final_flow, mask_0)
            # h_pred = homo_pred[-1]
            h_gt = batch['homography']
            four_points = [[0, 0],
                           [image_0.shape[-1]-1, 0],
                           [image_0.shape[-1]-1, image_0.shape[-2]-1],
                           [0, image_0.shape[-2]-1]]
            four_points = np.asarray(four_points, dtype=np.float32)
            mace = compute_mace(h_pred, h_gt, four_points)

            summary_writer.add_scalars('loss', {'train': loss.item()}, step)
            summary_writer.add_scalar('lr', scheduler.get_lr()[0], step)
            summary_writer.add_scalar('grad_norm', total_norm, step)
            summary_writer.add_scalars('mace', {'train': mace}, step)
            summary_writer.flush()
            print('Epoch: {} iter: {}/{} loss: {}, flow_loss: {}, homo_loss: {},mace: {}'.format(cur_epoch, iter_no + 1, steps_per_epoch, loss.item(), loss_flow, loss.item()-loss_flow , mace), flush=True)

            # # debug
            # break

    print('Epoch: {} loss: {}'.format(cur_epoch, np.mean(epoch_loss)))
    summary_writer.add_scalars('epoch_loss', {'train': np.mean(epoch_loss)}, cur_epoch)


def eval_one_epoch(model, data_loader, loss_fn, cur_epoch, steps_per_epoch, summary_writer):
    model.eval()
    epoch_loss = []
    epoch_mace = []
    with torch.no_grad():
        for iter_no, batch in tqdm(enumerate(data_loader), total=len(data_loader), disable=True):
            # evaluate 500 steps
            if iter_no>500:
                break

            image_0 = batch['image0']
            seg_0 = batch['seg0']
            image_1 = batch['image1']
            seg_1 = batch['seg1']
            # mask out non-planar regions
            image_0 = image_0 * seg_0[:, None, :, :]
            image_1 = image_1 * seg_1[:, None, :, :]

            mask_0 = batch['valid_mask0']
            # mask_0 = mask_0 * seg_0
            flow_gt = batch['flow']

            # model forward
            image_0 = image_0.cuda()
            image_1 = image_1.cuda()
            mask_0 = mask_0.cuda()
            flow_pred, homo_pred, residual = model(image_0, image_1, None, iters=5)
            # loss
            flow_gt = flow_gt.cuda()
            
            loss = 0.0
            loss_flow = 0.0  # only for output
            if loss_fn == 'sequence_loss':
                loss += sequence_loss(flow_pred, flow_gt, mask=mask_0, gamma=0.9)
            elif loss_fn == 'combined_loss':
                loss = sequence_loss(flow_pred, flow_gt, mask=mask_0, gamma=0.8)
                loss_flow = loss.item()
                # # compute corner_loss
                # four_points = torch.tensor([[0, 0], [W//8-1, 0], [W//8-1, H//8-1], [0, H//8-1]], dtype=torch.float32).cuda()
                # four_points = four_points.unsqueeze(0).repeat(B, 1, 1)
                # gt_h = batch['homography'].cuda()
                # loss += corner_loss(homo_pred, gt_h, four_points, gamma=0.9) * 0.1
                # compute residual loss
                loss += + residual_loss(residual, mask=mask_0, gamma=0.8) * 5
            else:
                assert False, 'loss_fn not supported'
            epoch_loss.append(loss.item())
            # compute homography and mace
            final_flow = flow_pred[-1]
            h_pred = compute_homography(final_flow, mask_0)
            h_gt = batch['homography']
            four_points = [[0, 0],
                           [image_0.shape[-1], 0],
                           [image_0.shape[-1], image_0.shape[-2]],
                           [0, image_0.shape[-2]]]
            four_points = np.asarray(four_points, dtype=np.float32)
            mace = compute_mace(h_pred, h_gt, four_points)
            epoch_mace.append(mace)

    summary_writer.add_scalars('loss', {'test': np.mean(epoch_loss)}, (cur_epoch + 1) * steps_per_epoch)
    summary_writer.add_scalars('mace', {'test': np.mean(epoch_mace)}, (cur_epoch + 1) * steps_per_epoch)
    summary_writer.flush()

    print('Epoch: {} val loss: {}'.format(cur_epoch, np.mean(epoch_loss)), flush=True)
    print('Epoch: {} val mean mace: {}'.format(cur_epoch, np.mean(epoch_mace)), flush=True)
    print('Epoch: {} val mdeian mace: {}'.format(cur_epoch, np.median(epoch_mace)), flush=True)


def do_train(model, train_loader, val_loader, optimizer, scheduler, loss_fn, checkpointer, checkpoint_arguments, config):
    log_dir = config['logging']['dir']
    summary_writer = SummaryWriter(log_dir)

    summary_writer.add_text('config', str(config), 0)
    
    epochs = config['trainer']['epochs']
    steps_per_epoch = config['trainer']['steps_per_epoch']
    start_epoch = checkpoint_arguments['step'] // steps_per_epoch

    for epoch in range(start_epoch, epochs):
        train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, epoch, steps_per_epoch, summary_writer)
        scheduler.step()

        eval_one_epoch(model, val_loader, loss_fn, epoch, steps_per_epoch, summary_writer)

        if epoch % config['trainer']['save_period'] == 0 or epoch == epochs - 1:
            checkpoint_arguments['step'] = (epoch+1) * steps_per_epoch
            checkpointer.save("model_{:02d}_epoch".format(epoch), **checkpoint_arguments)


def main(config_file_path):
    # Load yaml config file
    with open(config_file_path, 'r') as file:
        config = yaml.full_load(file)

    config['logging']['dir'] = config['logging']['dir'] + '-' + datetime.now().strftime('%y-%m-%d-%H_%M_%S')
    print("log dir: ", config['logging']['dir'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device: ', device)


    # load dataset and dataloader
    dataset_dir = config['data']['dir']
    train_list_path = os.path.join(dataset_dir, "train_scenes.txt")
    with open(train_list_path, "r") as f:
        train_scene_list = f.read().splitlines()
    test_list_path = os.path.join(dataset_dir, "test_scenes.txt")
    with open(test_list_path, "r") as f:
        test_scene_list = f.read().splitlines()
    image_size = config['data']['image_size']

    train_dataset = HypersimPlaneDataset(dataset_dir, train_scene_list, image_size)
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=8,
                              drop_last=True, collate_fn=collate_fn)
    val_dataset = HypersimPlaneDataset(dataset_dir, test_scene_list, image_size)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=4,
                            drop_last=True, collate_fn=collate_fn)
    # load model
    model = Raft().to(device)

    # load optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['trainer']['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['trainer']['milestones'],
                                                     gamma=config['trainer']['lr_decay'])
    loss_fn = config['trainer']['loss']
    # checkpoint
    checkpoint_arguments = {"step": 0}
    restart_lr = config['trainer']['restart_learning_rate'] if 'restart_learning_rate' in config[
        'trainer'] is not None else False
    optim_to_load = optimizer
    if restart_lr:
        optim_to_load = None
    checkpointer = CheckPointer(model, optim_to_load, scheduler, config['logging']['dir'], True, None,
                                device=device)
    extra_checkpoint_data = checkpointer.load()
    checkpoint_arguments.update(extra_checkpoint_data)

    # load pretrained model
    pretrained_model = config['model']['pretrained_model'] if 'pretrained_model' in config['model'] is not None else None
    if pretrained_model is not None:
        checkpoint = torch.load(pretrained_model, map_location=torch.device("cpu"))
        model_ = model
        model_.load_state_dict(checkpoint.pop("model"))
        print('Pretrained model loaded!')



    do_train(model, train_loader, val_loader, optimizer, scheduler, loss_fn, checkpointer, checkpoint_arguments, config)

    print('Training finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./configs/raft_hypersim.yaml')
    args = parser.parse_args()

    main(args.config_file)