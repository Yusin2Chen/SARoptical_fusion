import os
import math
import copy
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.datasets_oscd import OSCD_S2
from networks.ResUnet_cls import twinshift, MLPHead
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from utils.losses import HardNegtive_loss
from datasets.augmentation.augmentation import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, \
    RandomAffine, RandomPerspective
from datasets.augmentation.aug_params import RandomHorizontalFlip_params, RandomVerticalFlip_params, \
    RandomRotation_params, RandomAffine_params, RandomPerspective_params


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def get_scheduler(optimizer, args):
    if args.lr_step == "cos":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.epochs if args.T0 is None else args.T0,
            T_mult=args.Tmult,
            eta_min=args.eta_min,
        )
    elif args.lr_step == "step":
        m = [args.epochs - a for a in args.drop]
        return MultiStepLR(optimizer, milestones=m, gamma=args.drop_gamma)
    else:
        return None

def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    # 1600
    parser.add_argument('--batch_size', type=int, default=2000, help='batch_size')
    parser.add_argument('--crop_size', type=int, default=32, help='crop_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')

    # learning rate
    parser.add_argument("--T0", type=int, help="period (for --lr_step cos)")
    parser.add_argument("--Tmult", type=int, default=1, help="period factor (for --lr_step cos)")
    parser.add_argument("--lr_step", type=str, choices=["cos", "step", "none"], default="step",help="learning rate schedule type")
    parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    parser.add_argument("--eta_min", type=float, default=0, help="min learning rate (for --lr_step cos)")
    parser.add_argument("--adam_l2", type=float, default=1e-6, help="weight decay (L2 penalty)")
    parser.add_argument("--drop", type=int, nargs="*", default=[50, 25], help="milestones for learning rate decay (0 = last epoch)")
    parser.add_argument("--drop_gamma",type=float,default=0.2,help="multiplicative factor of learning rate decay")
    parser.add_argument("--no_lr_warmup", dest="lr_warmup", action="store_false", help="do not use learning rate warmup")

    # CLD related arguments
    parser.add_argument('--clusters', default=10, type=int, help='num of clusters for spectral clustering')
    parser.add_argument('--k-eigen', default=10, type=int, help='num of eigenvectors for k-way normalized cuts')
    parser.add_argument('--cld_t', default=0.07, type=float, help='temperature for spectral clustering')
    parser.add_argument('--use-kmeans', action='store_true', help='Whether use k-means for clustering. \
                            Use Normalized Cuts if it is False')
    parser.add_argument('--num-iters', default=20, type=int, help='num of iters for clustering')
    parser.add_argument('--Lambda', default=1.0, type=float, help='weight of mutual information loss')
    # resume path
    parser.add_argument('--resume', action='store_true', default=True, help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='resunet18', choices=['CMC_mlp3614','alexnet', 'resnet'])
    parser.add_argument('--in_dim', type=int, default=128, help='dim of feat for inner product')
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')

    # input/output
    parser.add_argument('--use_s2hr', action='store_true', default=True, help='use sentinel-2 high-resolution (10 m) bands')
    parser.add_argument('--use_s2mr', action='store_true', default=False, help='use sentinel-2 medium-resolution (20 m) bands')
    parser.add_argument('--use_s2lr', action='store_true', default=False, help='use sentinel-2 low-resolution (60 m) bands')
    parser.add_argument('--use_s1', action='store_true', default=True, help='use sentinel-1 data') #True for OSCD False for DFC2020
    parser.add_argument('--no_savanna', action='store_true', default=False, help='ignore class savanna')

    # add new views
    parser.add_argument('--data_dir_train', type=str, default='/workspace/OSCD_TS', help='path to training dataset')
    parser.add_argument('--dataset_val', type=str, default="dfc_cmc", choices=['sen12ms_holdout', '\
    dfc2020_val', 'dfc2020_test'], help='dataset to use for validation (default: sen12ms_holdout)')
    parser.add_argument('--model_path', type=str, default='./save_twins2s2_shift_spixTS', help='path to save model')
    parser.add_argument('--save', type=str, default='./save_twins2s2_shift_spixTS', help='path to save linear classifier')


    opt = parser.parse_args()

    # set up saving name
    opt.save_name = '{}_crop_{}_fetdim_{}'.format(opt.model, opt.crop_size, opt.feat_dim)
    opt.save_path = os.path.join(opt.save, opt.save_name)
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)

    if (opt.data_dir_train is None) or (opt.model_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')

    if not os.path.isdir(opt.dataset_val):
        os.makedirs(opt.dataset_val)

    if not os.path.isdir(opt.data_dir_train):
        raise ValueError('data path not exist: {}'.format(opt.data_dir_train))

    return opt

def get_train_loader(args):
    # load datasets
    train_set = OSCD_S2(args.data_dir_train,
                        subset="train",
                        no_savanna=args.no_savanna,
                        use_s2hr=args.use_s2hr,
                        use_s2mr=args.use_s2mr,
                        use_s2lr=args.use_s2lr,
                        use_s1=args.use_s1,
                        unlabeled=True,
                        transform=True,
                        train_index=None,
                        crop_size=args.crop_size)
    n_classes = train_set.n_classes
    n_inputs = train_set.n_inputs
    args.no_savanna = train_set.no_savanna
    args.display_channels = train_set.display_channels
    args.brightness_factor = train_set.brightness_factor

    # set up dataloaders
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)

    return train_loader, n_inputs, n_classes


class Trainer:
    def __init__(self, args, online_network, target_network, predictor, optimizer, scheduler, criterion, device):
        self.args = args
        self.augment_type = ['Horizontalflip', 'VerticalFlip']
        self.rot_agl = 15
        self.dis_scl = 0.2
        self.scl_sz = [0.8, 1.2]
        self.shear = [-0.2, 0.2]
        # self.mov_rg = random.uniform(-0.2, 0.2)
        self.aug_RHF = RandomHorizontalFlip(p=1)
        self.aug_RVF = RandomVerticalFlip(p=1)
        self.aug_ROT = RandomRotation(p=1, theta=self.rot_agl, interpolation='nearest')
        self.aug_PST = RandomPerspective(p=1, distortion_scale=0.3)
        self.aug_AFF = RandomAffine(p=1, theta=None, h_trans=random.uniform(0, 0.2), v_trans=random.uniform(0, 0.2),
                                    scale=None, shear=None, interpolation='nearest')
        self.online_network = online_network
        self.target_network = target_network
        self.predictor = predictor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.savepath = args.save_path
        self.criterion = criterion
        self.max_epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.feat_dim = args.feat_dim
        self.lr_warmup = args.lr_warmup_val
        self.lr = args.lr
        self.lr_step = args.lr_step

        for p in self.target_network.parameters():
            p.requires_grad = False

    def aug_list(self, img, model, params):
        for i in range(len(model)):
            img = model[i](img, params[i])
        return img

    def update_tau(self, step, max_steps):
        tau_upper, tau_lower = 1.0, 0.996
        self.tau = tau_upper - (tau_upper - tau_lower) * (math.cos(math.pi * step / max_steps) + 1) / 2

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.tau + param_q.data * (1. - self.tau)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_loader):

        niter = 0

        for epoch_counter in range(self.max_epochs):
            train_loss = 0.0
            iters = len(train_loader)
            for idx, batch in enumerate(train_loader):
                if self.lr_warmup < 50:
                    lr_scale = (self.lr_warmup + 1) / 50
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.lr * lr_scale
                    self.lr_warmup += 1

                image = batch['image']
                seb = batch['segments']
                ses = batch['segments_small']
                loss = self.update(image, seb, ses)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                niter += 1
                train_loss += loss.item()

                if self.lr_step == "cos" and self.lr_warmup >= 50:
                    self.scheduler.step(epoch_counter + idx / iters)
            if self.lr_step == "step":
                self.scheduler.step()
            train_loss = train_loss / len(train_loader)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch_counter, train_loss))
            # save checkpoints
            if (epoch_counter + 1) % 100 == 0:
                self.save_model(os.path.join(self.savepath, 'twins_epoch_{epoch}_{loss}.pth'.format(epoch=epoch_counter, loss=train_loss)))
            torch.cuda.empty_cache()

    def update(self, image, seb, ses):
        args = self.args
        sample_num = 1
        aug_type = random.sample(self.augment_type, sample_num)
        # augmentations
        model = []
        param = []
        if 'Horizontalflip' in aug_type:
            model.append(self.aug_RHF)
            param.append(RandomHorizontalFlip_params(0.5, image.shape[0], image.shape[-2:], self.device, image.dtype))
        if 'VerticalFlip' in aug_type:
            model.append(self.aug_RVF)
            param.append(RandomVerticalFlip_params(0.5, image.shape[0], image.shape[-2:], self.device, image.dtype))
        model.append(self.aug_AFF)
        param.append(RandomAffine_params(1.0, None, random.uniform(0.0, 0.2), random.uniform(0.0, 0.2),
                                         None, None, image.shape[0], image.shape[-2:], self.device, image.dtype))
        # split input
        batch_view_1, batch_view_2 = torch.split(image, [4, 4], dim=1)
        batch_view_1 = batch_view_1.to(self.device)
        batch_view_2 = batch_view_2.to(self.device)
        # tranforme one input view
        batch_view_1 = self.aug_list(batch_view_1, model, param)
        # center crop make sure no zero in input
        #16
        #aug_batch_view_1 = aug_batch_view_1[:, :, 8:24, 8:24]
        batch_view_1 = batch_view_1[:, :, 8:24, 8:24]
        batch_view_2 = batch_view_2[:, :, 8:24, 8:24]
        batch_segm_b = seb[:, 8: 24, 8: 24]
        batch_segm_s = ses[:, 8: 24, 8: 24]
        # compute query feature
        feature1 = self.online_network(batch_view_1, mode=0)
        feature2 = self.online_network(batch_view_2, mode=0)
        feature1 = self.predictor(feature1)
        feature2 = self.predictor(feature2)
        # alignment
        feature2 = self.aug_list(feature2, model, param)

        # compute key features
        with torch.no_grad():
            t_feature1  = self.target_network(batch_view_1, mode=0)
            t_feature2 = self.target_network(batch_view_2, mode=0)
            t_feature2 = self.aug_list(t_feature2, model, param)
            # generating_mask
            batch_segm_b = batch_segm_b.unsqueeze(dim=1)
            batch_segm_b = self.aug_list(batch_segm_b.float(), model, param)[:, 0, :, :]
            batch_segm_s = batch_segm_s.unsqueeze(dim=1)
            batch_segm_s = self.aug_list(batch_segm_s.float(), model, param)[:, 0, :, :]
            big_list, small_list = self.mask_spix(batch_segm_b, batch_segm_s)

        batch_segm_b = batch_segm_b.long().to(self.device)
        batch_segm_s = batch_segm_s.long().to(self.device)
        feature1, feature2 = self.avg_main_cls(batch_segm_b, big_list, feature1, feature2)
        t_feature1, t_feature2 = self.avg_main_cls(batch_segm_s, small_list, t_feature1, t_feature2)

        loss = self.criterion(feature1, t_feature2) + self.criterion(t_feature1, feature2)
        return loss


    def save_model(self, PATH):
        print('==> Saving...')
        state = {
            'online_network_state_dict': self.online_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(state, PATH)
        # help release GPU memory
        del state

    def avg_main_cls(self, batch_segm, batch_list, inFeats1, inFeats2):
        bs, C, H, W = inFeats1.shape
        outFeats1 = torch.zeros((bs, C)).to(self.device)
        outFeats2 = torch.zeros((bs, C)).to(self.device)

        for i in range(bs):
            s0 = batch_segm[i] == batch_list[i]
            ex_dim_s0 = s0[None, :, :]
            mask_nums = s0.sum(axis=0).sum(axis=0)
            masked1 = ex_dim_s0 * inFeats1[i]
            masked2 = ex_dim_s0 * inFeats2[i]
            ## first
            sum_sup_feats1 = masked1.sum(axis=1).sum(axis=1)
            avg_sup_feats1 = sum_sup_feats1 / mask_nums
            outFeats1[i, :] = avg_sup_feats1
            ## second
            sum_sup_feats2 = masked2.sum(axis=1).sum(axis=1)
            avg_sup_feats2 = sum_sup_feats2 / mask_nums
            outFeats2[i, :] = avg_sup_feats2

        return outFeats1, outFeats2

    def mask_spix(self, seb, ses):
        b, w, h = seb.shape
        big_list = []
        small_list = []
        for i in range(b):
            seb_i = seb[i].numpy().flatten().astype(np.uint16)
            seb_i = seb_i[seb_i != 0]
            idx = np.argmax(np.bincount(seb_i))
            big_list.append(idx)

            mask = seb[i] == idx
            ses_i = ses[i] * mask
            ses_i = ses_i.numpy().flatten().astype(np.uint16)
            ses_i = ses_i[ses_i != 0]
            small_list.append(np.argmax(np.bincount(ses_i)))

        return torch.tensor(big_list).long().to(self.device), torch.tensor(small_list).long().to(self.device)

    def unique(self, x, dim=0):
        unique, inverse = torch.unique(
            x, sorted=True, return_inverse=True, dim=dim)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                            device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

def main():

    # parse the args
    args = parse_option()

    # set flags for GPU processing if available
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda'

    # set the data loader
    train_loader, n_inputs, n_classes = get_train_loader(args)
    args.n_inputs = n_inputs
    args.n_classes = n_classes

    # set the model
    online_network = twinshift(width=1, in_channel=4, in_dim=args.in_dim, feat_dim=args.feat_dim).to(device)
    target_network = copy.deepcopy(online_network)
    target_network = target_network.to(device)
    # predictor network
    predictor = MLPHead(args.feat_dim, int(args.feat_dim * 1.5), args.feat_dim).to(device)
    #--> optimizer
    optimizer = torch.optim.Adam(list(online_network.parameters()) + list(predictor.parameters()), lr=3e-4, weight_decay=1e-4)
    scheduler = get_scheduler(optimizer, args)
    args.lr_warmup_val = 0 if args.lr_warmup else 50
    criterion = HardNegtive_loss()
    trainer = Trainer(args,
                      online_network=online_network,
                      target_network=target_network,
                      predictor=predictor,
                      optimizer=optimizer,
                      criterion=criterion,
                      scheduler=scheduler,
                      device=device)

    trainer.train(train_loader)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()

