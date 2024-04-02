from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
from utils import *  # Assumes utility functions and classes are defined here
from model import *  # Assumes the model definition is here
import time
import os
import torch.nn.functional as F
import numpy as np

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=2, help="Factor to scale the image resolution")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use for computation")
    parser.add_argument('--batch_size', type=int, default=36, help="Batch size for training")
    parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch number')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='Steps after which to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='./data/train', help='Directory for the training dataset')
    parser.add_argument('--model_name', type=str, default='EISR', help='Model name for saving/loading')
    parser.add_argument('--load_pretrain', type=bool, default=False, help='Whether to load a pretrained model')
    parser.add_argument('--model_path', type=str, default='log/EISR.pth.tar', help='Path to the model file')
    return parser.parse_args()

# Train the model
def train(train_loader, cfg):
    net = Net(cfg.scale_factor).to(cfg.device)  # Initialize the model
    cudnn.benchmark = True  # Enable cudnn benchmark for faster computation
    scale = cfg.scale_factor

    # Load pre-trained model if specified
    if cfg.load_pretrain:
        if os.path.isfile(cfg.model_path):
            model = torch.load(cfg.model_path, map_location={'cuda:0': cfg.device})
            net.load_state_dict(model['state_dict'])
            cfg.start_epoch = model["epoch"]
        else:
            print("=> no model found at '{}'".format(cfg.model_path))

    criterion_L1 = torch.nn.L1Loss().to(cfg.device)  # L1 loss for training
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad], lr=cfg.lr)  # Adam optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)  # Learning rate scheduler

    for idx_epoch in range(cfg.start_epoch, cfg.n_epochs):
        start_time = time.time()
        loss_epoch = []  # To track loss per epoch
        psnr_epoch = []  # To track PSNR per epoch
        loss_list = []
        psnr_list = []        
        for idx_iter, (HR_left, HR_right, LR_left, LR_right) in enumerate(train_loader):
            
            b, c, h, w = LR_left.shape
            HR_left, HR_right, LR_left, LR_right  = Variable(HR_left).to(cfg.device), Variable(HR_right).to(cfg.device),\
                                                    Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)

            SR_left, SR_right, (M_right_to_left, M_left_to_right), (V_left, V_right)\
                = net(LR_left, LR_right, is_training=1)

            ''' SR Loss '''
            loss_SR = criterion_L1(SR_left, HR_left) + criterion_L1(SR_right, HR_right)

            ''' Photometric Loss '''
            Res_left = torch.abs(HR_left - F.interpolate(LR_left, scale_factor=scale, mode='bicubic', align_corners=False))
            Res_left = F.interpolate(Res_left, scale_factor=1 / scale, mode='bicubic', align_corners=False)
            Res_right = torch.abs(HR_right - F.interpolate(LR_right, scale_factor=scale, mode='bicubic', align_corners=False))
            Res_right = F.interpolate(Res_right, scale_factor=1 / scale, mode='bicubic', align_corners=False)
            Res_leftT = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_right.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                  ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_rightT = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_left.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                   ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            loss_photo = criterion_L1(Res_left * V_left.repeat(1, 3, 1, 1), Res_leftT * V_left.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right * V_right.repeat(1, 3, 1, 1), Res_rightT * V_right.repeat(1, 3, 1, 1))

            ''' Smoothness Loss '''
            loss_h = criterion_L1(M_right_to_left[:, :-1, :, :], M_right_to_left[:, 1:, :, :]) + \
                     criterion_L1(M_left_to_right[:, :-1, :, :], M_left_to_right[:, 1:, :, :])
            loss_w = criterion_L1(M_right_to_left[:, :, :-1, :-1], M_right_to_left[:, :, 1:, 1:]) + \
                     criterion_L1(M_left_to_right[:, :, :-1, :-1], M_left_to_right[:, :, 1:, 1:])
            loss_smooth = loss_w + loss_h

            ''' Cycle Loss '''
            Res_left_cycle = torch.bmm(M_right_to_left.contiguous().view(b * h, w, w), Res_rightT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                       ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            Res_right_cycle = torch.bmm(M_left_to_right.contiguous().view(b * h, w, w), Res_leftT.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                        ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            loss_cycle = criterion_L1(Res_left * V_left.repeat(1, 3, 1, 1), Res_left_cycle * V_left.repeat(1, 3, 1, 1)) + \
                         criterion_L1(Res_right * V_right.repeat(1, 3, 1, 1), Res_right_cycle * V_right.repeat(1, 3, 1, 1))

            ''' Consistency Loss '''
            SR_left_res = F.interpolate(torch.abs(HR_left - SR_left), scale_factor=1 / scale, mode='bicubic', align_corners=False)
            SR_right_res = F.interpolate(torch.abs(HR_right - SR_right), scale_factor=1 / scale, mode='bicubic', align_corners=False)
            SR_left_resT = torch.bmm(M_right_to_left.detach().contiguous().view(b * h, w, w), SR_right_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                     ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            SR_right_resT = torch.bmm(M_left_to_right.detach().contiguous().view(b * h, w, w), SR_left_res.permute(0, 2, 3, 1).contiguous().view(b * h, w, c)
                                      ).view(b, h, w, c).contiguous().permute(0, 3, 1, 2)
            loss_cons = criterion_L1(SR_left_res * V_left.repeat(1, 3, 1, 1), SR_left_resT * V_left.repeat(1, 3, 1, 1)) + \
                       criterion_L1(SR_right_res * V_right.repeat(1, 3, 1, 1), SR_right_resT * V_right.repeat(1, 3, 1, 1))

            ''' Total Loss '''
            loss = loss_SR + 0.1 * loss_cons + 0.1 * (loss_photo + loss_smooth + loss_cycle)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            psnr_epoch.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left[:,:,:,64:].data.cpu()))
            loss_epoch.append(loss.data.cpu())
        end_time = time.time()
        epoch_time = end_time - start_time
        scheduler.step()

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))

            print('Epoch--%4d, loss--%f, loss_SR--%f, loss_photo--%f, loss_smooth--%f, loss_cycle--%f, loss_cons--%f,PSNR----%f, Time--%.2fs' %
                  (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(loss_SR.data.cpu()).mean()),
                   float(np.array(loss_photo.data.cpu()).mean()), float(np.array(loss_smooth.data.cpu()).mean()),
                   float(np.array(loss_cycle.data.cpu()).mean()), float(np.array(loss_cons.data.cpu()).mean()), float(np.array(psnr_epoch).mean()),epoch_time))
            torch.save({'epoch': idx_epoch + 1, 'state_dict': net.state_dict()},
                       'log/' + cfg.model_name + '_' + str(cfg.scale_factor) + 'xSR_epoch' + str(idx_epoch + 1) + '.pth.tar')
            loss_epoch = []


# Main function to load data and train the model
def main(cfg):
    train_set = TrainSetLoader(cfg)  # Load the training dataset
    train_loader = DataLoader(dataset=train_set, num_workers=6, batch_size=cfg.batch_size, shuffle=True)  # DataLoader for the dataset
    train(train_loader, cfg)  # Start training

if __name__ == '__main__':
    cfg = parse_args()  # Parse command line arguments
    main(cfg)  # Start the main function

