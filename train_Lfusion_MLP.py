import os
import argparse
import torch
from torch.utils.data import DataLoader
from datasets.dataset_dfc import DFC2020
from networks.propnets import L_Fusion
from networks.linear_eva import MLP
import utils.metrics as metrics
from utils.util import adjust_learning_rate, AverageMeter, accuracy, seed_torch, visualization


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    # 1600
    parser.add_argument('--batch_size', type=int, default=10, help='batch_size')
    parser.add_argument('--crop_size', type=int, default=64, help='crop_size')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('--in_dim', type=int, default=128, help='dim of feat for inner product')
    parser.add_argument('--feat_dim', type=int, default=128, help='dim of feat for inner product')
    # resume path
    parser.add_argument('--resume', action='store_true', default=False, help='path to latest checkpoint (default: none)')
    parser.add_argument('--valid', action='store_true', default=True, help='for validation')

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
    parser.add_argument('--save', type=str, default='./mlp_Lfusion1000', help='path to save linear classifier')
    parser.add_argument('--save_freq', type=int, default=50, help='number of training epochs')
    parser.add_argument('--preview_dir', type=str, default='./preview_Lfusion',
                        help='path to preview dir (default: no previews)')

    opt = parser.parse_args()

    # set up saving name
    if not os.path.isdir(opt.save):
        os.makedirs(opt.save)

    if (opt.data_dir_train is None) or (opt.model_path is None):
        raise ValueError('one or more of the folders is None: data_folder | model_path | tb_path')


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
                        transform=False,
                        unlabeled=False,
                        train_index='./utils/train1000.npy')
    n_classes = train_set.n_classes
    n_inputs = train_set.n_inputs
    args.no_savanna = train_set.no_savanna
    args.display_channels = train_set.display_channels
    args.brightness_factor = train_set.brightness_factor

    valid_set = DFC2020(args.data_dir_train,
                        subset="train",
                        no_savanna=args.no_savanna,
                        use_s2hr=args.use_s2hr,
                        use_s2mr=args.use_s2mr,
                        use_s2lr=args.use_s2lr,
                        use_s1=args.use_s1,
                        transform=False,
                        unlabeled=False,
                        train_index='./utils/vali5114.npy')
    # set up dataloaders
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)

    valid_loader = DataLoader(valid_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True,
                              drop_last=False)
    return train_loader, valid_loader, n_inputs, n_classes


def unet_encoder_factory(args, pretrained=True):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        input_channels: the number of output channels
    """
    # build model
    model = L_Fusion(width=0.5, in_dim=args.in_dim, feat_dim=args.feat_dim).to(args.device)
    if pretrained:
        # load pre-trained model
        print('==> loading pre-trained model')
        ckpt = torch.load('./save/Lfusion/twins_epoch_699_13.25080394744873.pth')
        pretrained_dict = ckpt['online_network_state_dict']
        model_dict = model.state_dict()
        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def train(epoch, train_loader, online_network, classifier, criterion, optimizer, args):
    """
    one epoch training
    """
    # set model to train mode
    online_network.eval()
    classifier.train()


    for idx, (batch) in enumerate(train_loader):

        # unpack sample
        image, target = batch['image'], batch['label']
        #image = image[:, :, 64:192, 64:192]
        #target = target[:, 64:192, 64:192]
        if args.use_gpu:
            image, target = image.cuda(), target.cuda()
        batch_view_1, batch_view_2 = torch.split(image, [4, 2], dim=1)
        # ===================forward=====================
        with torch.no_grad():
            feature1, feature2 = online_network(batch_view_1, batch_view_2, mode=1)
        prediction = classifier(torch.cat((feature1, feature2), 1))
        loss = criterion(prediction, target)

        # ===================backward=====================
        # reset gradients
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()


        # print info
        print(f'\rtrain loss : {loss.item():.5f}| step :{idx}/{len(train_loader)}|{epoch}', end='', flush=True)



def validate(val_loader, online_network, classifier, criterion, args):
    """
    evaluation
    """
    # switch to evaluate mode
    online_network.eval()
    classifier.eval()

    # main validation loop
    loss = 0
    conf_mat = metrics.ConfMatrix(args.n_classes)

    with torch.no_grad():
        for idx, (batch) in enumerate(val_loader):

            # unpack sample
            image, target = batch['image'], batch['label']
            #image = image[:, :, 64:192, 64:192]
            #target = target[:, 64:192, 64:192]
            if args.use_gpu:
                image, target = image.cuda(), target.cuda()
            batch_view_1, batch_view_2 = torch.split(image, [4, 2], dim=1)
            # ===================forward=====================
            with torch.no_grad():
                feature1, feature2 = online_network(batch_view_1, batch_view_2, mode=1)
            prediction = classifier(torch.cat((feature1, feature2), 1))
            loss += criterion(prediction, target).cpu().item()

            # calculate error metrics
            conf_mat.add_batch(target, prediction.max(1)[1])
            if args.visual:
                visualization(prediction, target, batch['id'], image, args)

        print("[Val] AA: {:.2f}%".format(conf_mat.get_aa() * 100))
        #print("[Val] mIoU: ", conf_mat.get_mIoU())
        if args.valid:
            print("[Val] SA: ", conf_mat.get_sa())
            print("[Val] mIoU: ", conf_mat.get_mIoU())
            print("[Val] IoU: ", conf_mat.get_cf())

def main():
    # set seed
    seed_torch(seed=0)
    # parse the args
    args = parse_option()

    # set flags for GPU processing if available
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        args.use_gpu = True
    else:
        args.use_gpu = False
    # set the data loader
    train_loader, valid_loader, n_inputs, n_classes = get_train_loader(args)
    args.n_inputs = n_inputs
    args.n_classes = n_classes

    # set the model
    online_network = unet_encoder_factory(args, pretrained=True)
    # predictor network
    classifier = MLP(256, n_classes).to(args.device)
    args.visual = True
    if args.valid:
        try:
            print('==>loading pretrained Linear model')
            checkpoints_folder = os.path.join('.', args.save)
            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'ckpt_epoch_50.pth')),
                                     map_location=args.device)
            classifier.load_state_dict(load_params['classifier'])
        except FileNotFoundError:
            print("Pre-trained weights not found. Please to check.")
    # target encoder
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    # train and valid
    args.start_epoch = 1
    for epoch in range(args.start_epoch, args.epochs + 1):

        if not args.valid:
            # adjust_learning_rate(epoch, args, optimizer)
            train(epoch, train_loader, online_network, classifier, criterion, optimizer, args)
            if epoch % 50 == 0:
                validate(valid_loader, online_network, classifier, criterion, args)
        else:
            validate(valid_loader, online_network, classifier, criterion, args)
            break

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'epoch': epoch,
                'classifier': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_name = 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch)
            save_name = os.path.join(args.save, save_name)
            print('saving regular model!')
            torch.save(state, save_name)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()

