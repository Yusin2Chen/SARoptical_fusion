import os
import argparse
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from datasets.dataset_dfc import DFC2020
from networks.propnets import E_Fusion
from utils.util import RandomApply, default, seed_torch
from utils.losses import HardNegtive_loss
from kornia import filters
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingWarmRestarts
from utils.augmentation.augmentation import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, \
    RandomAffine, RandomPerspective
from utils.augmentation.aug_params import RandomHorizontalFlip_params, RandomVerticalFlip_params, \
    RandomRotation_params, RandomAffine_params, RandomPerspective_params

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
    parser.add_argument('--batch_size', type=int, default=1000, help='batch_size')
    parser.add_argument('--crop_size', type=int, default=32, help='crop_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=700, help='number of training epochs')

    # resume path
    parser.add_argument('--resume', action='store_true', default=False, help='path to latest checkpoint (default: none)')
    parser.add_argument('--in_dim', type=int, default=256, help='dim of feat for inner product')
    parser.add_argument('--feat_dim', type=int, default=256, help='dim of feat for inner product')

    # learning rate
    parser.add_argument("--T0", type=int, help="period (for --lr_step cos)")
    parser.add_argument("--Tmult", type=int, default=1, help="period factor (for --lr_step cos)")
    parser.add_argument("--lr_step", type=str, choices=["cos", "step", "none"], default="step",
                        help="learning rate schedule type")
    parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    parser.add_argument("--eta_min", type=float, default=0, help="min learning rate (for --lr_step cos)")
    parser.add_argument("--adam_l2", type=float, default=1e-6, help="weight decay (L2 penalty)")
    parser.add_argument("--drop", type=int, nargs="*", default=[50, 25],
                        help="milestones for learning rate decay (0 = last epoch)")
    parser.add_argument("--drop_gamma", type=float, default=0.2, help="multiplicative factor of learning rate decay")
    parser.add_argument("--no_lr_warmup", dest="lr_warmup", action="store_false",
                        help="do not use learning rate warmup")

    # input/output
    parser.add_argument('--use_s2hr', action='store_true', default=True, help='use sentinel-2 high-resolution (10 m) bands')
    parser.add_argument('--use_s2mr', action='store_true', default=False, help='use sentinel-2 medium-resolution (20 m) bands')
    parser.add_argument('--use_s2lr', action='store_true', default=False, help='use sentinel-2 low-resolution (60 m) bands')
    parser.add_argument('--use_s1', action='store_true', default=True, help='use sentinel-1 data') #True for OSCD False for DFC2020
    parser.add_argument('--no_savanna', action='store_true', default=False, help='ignore class savanna')

    # add new views
    #'/workplace/OSCD'
    #'/R740-75T/Chenyx/Workplace/OSCD'
    parser.add_argument('--data_dir_train', type=str, default='/workplace/DFC2020', help='path to training dataset')
    parser.add_argument('--model_path', type=str, default='./save', help='path to save model')
    parser.add_argument('--save', type=str, default='./EfusionS2', help='path to save linear classifier')


    opt = parser.parse_args()

    # set up saving name
    opt.save_path = os.path.join(opt.model_path, opt.save)
    if not os.path.isdir(opt.save_path):
        os.makedirs(opt.save_path)

    if not os.path.isdir(opt.data_dir_train):
        raise ValueError('data path not exist: {}'.format(opt.data_dir_train))

    return opt

def get_train_loader(args):
    # load datasets
    train_set = DFC2020(args.data_dir_train,
                        subset="train",
                        no_savanna=args.no_savanna,
                        use_s2hr=args.use_s2hr,
                        use_s2mr=args.use_s2mr,
                        use_s2lr=args.use_s2lr,
                        use_s1=args.use_s1,
                        transform=True,
                        unlabeled=True,
                        crop_size=args.crop_size)
                        #train_index='./utils/train_40.npy')
    n_classes = train_set.n_classes
    n_inputs = train_set.n_inputs
    args.no_savanna = train_set.no_savanna
    args.display_channels = train_set.display_channels
    args.brightness_factor = train_set.brightness_factor

    train_size = int(0.16 * len(train_set))
    test_size = len(train_set) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_set, [train_size, test_size])

    # set up dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)

    return train_loader, n_inputs, n_classes


class Trainer:
    def __init__(self, args, online_network, optimizer, criterion, scheduler, device):

        self.args = args
        DEFAULT_AUG = RandomApply(filters.GaussianBlur2d((11, 11), (1.5, 2.5)), p=0.5)
        augment_fn = None
        self.augment = default(augment_fn, DEFAULT_AUG)
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
        self.aug_AFF = RandomAffine(p=1, theta=0, h_trans=random.uniform(-0.2, 0.2), v_trans=random.uniform(-0.2, 0.2),
                                    scale=None, shear=None, interpolation='nearest')
        self.online_network = online_network
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

    def aug_list(self, img, model, params):
        for i in range(len(model)):
            img = model[i](img, params[i])
        return img

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
                segmt = batch['segments']
                loss = self.update(image, segmt)

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

    def update(self, image, segmt):
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
        param.append(RandomAffine_params(1.0, 0.0, random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2),
                                         None, None, image.shape[0], image.shape[-2:], self.device, image.dtype))
        # split input
        image, _ = torch.split(image, [4, 2], dim=1)
        batch_view_1 = image.to(self.device)
        batch_view_2 = image.to(self.device)
        # tranforme one input view
        batch_view_1 = self.aug_list(batch_view_1, model, param)
        # 32
        batch_view_1 = batch_view_1[:, :, 8: 24, 8: 24]
        batch_view_2 = batch_view_2[:, :, 8: 24, 8: 24]
        batch_segm_2 = segmt[:, 8: 24, 8: 24]
        # compute query feature
        l_feature1, l_feature2, loss_vq = self.online_network(batch_view_1, batch_view_2, mode=0)
        l_feature2 = self.aug_list(l_feature2, model, param)

        # mask no-overlap
        with torch.no_grad():
            batch_segm_2 = batch_segm_2.unsqueeze(dim=1)
            batch_segm_2 = self.aug_list(batch_segm_2.float(), model, param)[:, 0, :, :]
            ones = self.mask_spix(batch_segm_2)
            one_mask = ones.long().eq(1).to(self.device)

        batch_segm_2 = batch_segm_2.long().to(self.device)
        l_feature1, l_feature2= self.get_spix_data(batch_segm_2, one_mask, l_feature1, l_feature2)
        # pixel loss
        loss = self.criterion(l_feature1, l_feature2) + loss_vq


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

    def get_spix_data(self, batch_segm_2, one_mask, inFeats1, inFeats2):
        bs, C, H, W = inFeats1.shape
        batch_segm_2 = batch_segm_2 * one_mask
        one_mask = one_mask.contiguous().view(-1)
        new_seg = batch_segm_2.view(-1).contiguous()
        values_idx = new_seg[one_mask]
        outFeats1 = torch.zeros((len(values_idx), C)).to(self.device)
        outFeats2 = torch.zeros((len(values_idx), C)).to(self.device)
        unique = torch.unique(values_idx, sorted=False, return_inverse=False, dim=0)

        for i in unique:
            s0 = batch_segm_2 == i
            s0_idx = values_idx == i
            spix_idx = s0.sum(axis=1).sum(axis=1) == 0
            ex_dim_s0 = s0[:, None, :, :]
            mask_nums = s0.sum(axis=1).sum(axis=1)
            mask_nums[mask_nums == 0] = 1
            mask_nums = mask_nums[:, None]
            masked1 = ex_dim_s0 * inFeats1
            masked2 = ex_dim_s0 * inFeats2
            ## first
            sum_sup_feats1 = masked1.sum(axis=2).sum(axis=2)
            avg_sup_feats1 = sum_sup_feats1 / mask_nums
            outFeats1[s0_idx, :] = avg_sup_feats1[~spix_idx, :]
            ## second
            sum_sup_feats2 = masked2.sum(axis=2).sum(axis=2)
            avg_sup_feats2 = sum_sup_feats2 / mask_nums
            outFeats2[s0_idx, :] = avg_sup_feats2[~spix_idx, :]

        return outFeats1, outFeats2

    def mask_spix(self, image):
        b, w, h = image.shape
        zero = torch.zeros((b, w, h))
        samples = np.random.randint(w, size=(200, 2))

        for i in range(b):
            img_i = image[i][samples[:, 0], samples[:, 1]]
            val_i, index = self.unique(img_i)
            if len(val_i) > 0 and val_i[0] == 0:
                val_i = val_i[1::]
                index = index[1::]
            # print(val_i)
            if len(index) == 1:
                unique_i = samples[index]
                zero[i][unique_i[0], unique_i[1]] = 1
            elif len(index) > 1:
                unique_i = samples[index]
                zero[i][unique_i[:, 0], unique_i[:, 1]] = 1
        return zero

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
    online_network = E_Fusion(width=1, in_channel=4, in_dim=args.in_dim, feat_dim=args.feat_dim).to(device)
    ## load pre-trained model if defined
    if args.resume:
        try:
            print('loading pretrained models')
            checkpoints_folder = os.path.join('.', 'save/Efusion_encoder')

            # load pre-trained parameters
            load_params = torch.load(
                os.path.join(os.path.join(checkpoints_folder, 'twins_epoch_199_4.75602936.pth')),
                map_location=device)

            online_network.load_state_dict(load_params['online_network_state_dict'], strict=False)

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # target encoder
    criterion = HardNegtive_loss()
    optimizer = torch.optim.Adam(online_network.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = get_scheduler(optimizer, args)
    args.lr_warmup_val = 0 if args.lr_warmup else 50
    trainer = Trainer(args,
                      online_network=online_network,
                      optimizer=optimizer,
                      criterion=criterion,
                      scheduler=scheduler,
                      device=device)

    trainer.train(train_loader)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    seed_torch(seed=1024)
    main()

