import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets.dataset_dfc import DFC2020
from utils.losses import CCA_loss
from utils.util import RandomApply, default, seed_torch
from utils.augmentation.augmentation import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, \
    RandomAffine, RandomPerspective
from utils.augmentation.aug_params import RandomHorizontalFlip_params, RandomVerticalFlip_params, \
    RandomRotation_params, RandomAffine_params, RandomPerspective_params
import os
from functools import partial
import argparse
from networks.DCCA import DCCA



def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    # 1600
    parser.add_argument('--batch_size', type=int, default=1000, help='batch_size')
    parser.add_argument('--crop_size', type=int, default=32, help='crop_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=700, help='number of training epochs')

    # resume path
    parser.add_argument('--resume', action='store_true', default=False, help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='resunet18', choices=['CMC_mlp3614','alexnet', 'resnet'])
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')
    parser.add_argument('--modal', default='CMC', type=str, choices=['RGB', 'CMC'],
                        help='single RGB modal, or two modalities in CMC')
    parser.add_argument('--jigsaw', action='store_true', default=False, help='adding PIRL branch')
    parser.add_argument('--mem', default='bank', type=str, choices=['bank', 'moco'],
                        help='memory mechanism: memory bank, or moco encoder cache')
    parser.add_argument('--arch', default='resnet18', type=str, help='e.g., resnet50, resnext50')
    parser.add_argument('--head', default='linear', type=str, choices=['linear', 'mlp'], help='projection head')

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
    parser.add_argument('--dataset_val', type=str, default="dfc_cmc", choices=['sen12ms_holdout', '\
    dfc2020_val', 'dfc2020_test'], help='dataset to use for validation (default: sen12ms_holdout)')
    parser.add_argument('--model_path', type=str, default='./save', help='path to save model')
    parser.add_argument('--save', type=str, default='./DCCA', help='path to save linear classifier')


    opt = parser.parse_args()

    # set up saving name
    opt.save_path = os.path.join(opt.model_path, opt.save)
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
    def __init__(self, args, model, optimizer, objective, lr_scheduler, device):

        self.augment_type = ['Horizontalflip', 'VerticalFlip', 'Affine']
        self.rot_agl = 30
        self.dis_scl = 0.2
        self.scl_sz = [0.8, 1.2]
        self.shear = [-0.2, 0.2]
        self.aug_RHF = RandomHorizontalFlip(p=1)
        self.aug_RVF = RandomVerticalFlip(p=1)
        self.aug_ROT = RandomRotation(p=1, theta=self.rot_agl)
        self.aug_PST = RandomPerspective(p=1, distortion_scale=0.3)
        self.aug_AFF = RandomAffine(p=1, theta=self.rot_agl, h_trans=0.0, v_trans=0.0, scale=self.scl_sz, shear=self.shear)

        self.model = model
        self.objective = objective
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.savepath = args.save_path
        self.max_epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers


    def aug_list(self, img, model, params):
        for i in range(len(model)):
            img = model[i](img, params[i])
        return img


    def train(self, train_loader):

        niter = 0

        for epoch_counter in range(self.max_epochs):
            train_loss = 0.0
            for idx, batch in enumerate(train_loader):
                image = batch['image']
                loss = self.update(image)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #self.lr_scheduler.step()

                niter += 1
                train_loss += loss.item()

            train_loss = train_loss / len(train_loader)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch_counter, train_loss))
            # save checkpoints
            if (epoch_counter + 1) % 100 == 0:
                self.save_model(os.path.join(self.savepath, 'DCCA_epoch_{epoch}_{loss}.pth'.format(epoch=epoch_counter, loss=train_loss)))
            torch.cuda.empty_cache()

    def update(self, image):
        sample_num = 1
        aug_type = random.sample(self.augment_type, sample_num)
        model = []
        param = []
        if 'Horizontalflip' in aug_type:
            model.append(self.aug_RHF)
            param.append(RandomHorizontalFlip_params(0.5, image.shape[0], image.shape[-2:], self.device, image.dtype))
        if 'VerticalFlip' in aug_type:
            model.append(self.aug_RVF)
            param.append(RandomVerticalFlip_params(0.5, image.shape[0], image.shape[-2:], self.device, image.dtype))
        if 'Affine' in aug_type:
            model.append(self.aug_AFF)
            param.append(RandomAffine_params(1.0, 0.0, random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2),
                                             None, None, image.shape[0], image.shape[-2:], self.device,
                                             image.dtype))

        image = self.aug_list(image, model, param).to(self.device)
        # center crop make sure no zero in input
        image = image[:, :, 8:24, 8:24]
        # compute query feature
        predictions_from_view_1, predictions_from_view_2 = self.model(image, mode=1)
        # loss calculation
        loss = self.objective(F.normalize(predictions_from_view_1, dim=1), F.normalize(predictions_from_view_2, dim=1))
        return loss

    def save_model(self, PATH):
        print('==> Saving...')
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(state, PATH)
        # help release GPU memory
        del state


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
    model = DCCA(n_inputs=n_inputs).to(device)
    ## load pre-trained model if defined
    if args.resume:
        try:
            print('loading pretrained models')
            checkpoints_folder = os.path.join('.', 'pre_train')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'BYOL' + str(args.crop_size) + '.pth')),
                                     map_location=device)

            model.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")


    # target encoder
    objective = partial(CCA_loss, outdim_size=10, use_all_singular_values=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
    trainer = Trainer(args, model, optimizer, objective, lr_scheduler, device)

    trainer.train(train_loader)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    seed_torch(seed=1024)
    main()

