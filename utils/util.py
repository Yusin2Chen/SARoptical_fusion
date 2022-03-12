from __future__ import print_function
import os
import torch
import random
from torch import nn
import numpy as np
from matplotlib import colors
import matplotlib.patches as mpatches
from PIL import Image
import matplotlib.pyplot as plt

# standar
S2_MEAN = np.array([1353.3418, 1265.4015, 1269.009, 1976.1317])
S2_STD  = np.array([242.07303, 290.84450, 402.9476, 516.77480])

def Rnormalize_S2(imgs):
    for i in range(4):
        imgs[:,i,:,:] = (imgs[:,i,:,:] * S2_STD[i]) + S2_MEAN[i]
    return imgs


S1_MEAN = np.array([-9.020017, -15.73008])
S1_STD  = np.array([3.5793820, 3.671725])

def Rnormalize_S1(imgs):
    for i in range(2):
        imgs[:,i,:,:] = (imgs[:,i,:,:] * S1_STD[i]) + S1_MEAN[i]
    return imgs

def default(val, def_val):
    return def_val if val is None else val

# augmentation utils

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def convert_to_np(tensor):
    # convert pytorch tensors to numpy arrays
    if not isinstance(tensor, np.ndarray):
        tensor = tensor.cpu().numpy()
    return tensor


def labels_to_dfc(tensor, no_savanna):
    """
    INPUT:
    Classes encoded in the training scheme (0-9 if savanna is a valid label
    or 0-8 if not). Invalid labels are marked by 255 and will not be changed.

    OUTPUT:
    Classes encoded in the DFC2020 scheme (1-10, and 255 for invalid).
    """

    # transform to numpy array
    tensor = convert_to_np(tensor)

    # copy the original input
    out = np.copy(tensor)

    # shift labels if there is no savanna class
    if no_savanna:
        for i in range(2, 9):
            out[tensor == i] = i + 1
    else:
        pass

    # transform from zero-based labels to 1-10
    out[tensor != 255] += 1

    # make sure the mask is intact and return transformed labels
    assert np.all((tensor == 255) == (out == 255))
    return out


def display_input_batch(tensor, display_indices=0, brightness_factor=3):

    # extract display channels
    tensor = tensor[:, display_indices, :, :]

    # restore NCHW tensor shape if single channel image
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)

    # scale image
    tensor = torch.clamp((tensor * brightness_factor), 0, 1)

    return tensor


def display_label_batch(tensor, no_savanna=False):

    # get predictions if input is one-hot encoded
    if len(tensor.shape) == 4:
        tensor = tensor.max(1)[1]

    # convert train labels to DFC2020 class scheme
    tensor = labels_to_dfc(tensor, no_savanna)

    # colorize labels
    cmap = mycmap()
    imgs = []
    for s in range(tensor.shape[0]):
        im = (tensor[s, :, :] - 1) / 10
        im = cmap(im)[:, :, 0:3]
        im = np.rollaxis(im, 2, 0)
        imgs.append(im)
    tensor = np.array(imgs)

    return tensor


def classnames():
    return ["Forest", "Shrubland", "Savanna", "Grassland", "Wetlands",
            "Croplands", "Urban/Built-up", "Snow/Ice", "Barren", "Water"]


def mycmap():
    cmap = colors.ListedColormap(['#009900',
                                  '#c6b044',
                                  '#fbff13',
                                  '#b6ff05',
                                  '#27ff87',
                                  '#c24f44',
                                  '#a5a5a5',
                                  '#69fff8',
                                  '#f9ffa4',
                                  '#1c0dff',
                                  '#ffffff'])
    return cmap

def newcmap():
    cmap = colors.ListedColormap(['#009900',  # 0
                                  '#c6b044',  # 1
                                  '#fbff13',  # 2
                                  '#27ff87',  # 3
                                  '#69fff8',  # 4
                                  '#1c0dff',  # 5
                                  '#ffffff',  # 6
                                  '#69fff8',  # 7
                                  '#f9ffa4',  # 8
                                  '#1c0dff',  # 9
                                  '#ffffff'])  # 10
    return cmap

def mypatches():
    patches = []
    for counter, name in enumerate(classnames()):
        patches.append(mpatches.Patch(color=mycmap().colors[counter],
                                      label=name))
    return patches

def seed_torch(seed=1029):
    random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def visualizationS1(prediction, target, ID, image, args): # ID-> batch['id']
    id_list = ['ROIs0000_autumn_dfc_BandarAnzali_p611.tif', 'ROIs0000_autumn_dfc_CapeTown_p1277.tif',
               'ROIs0000_autumn_dfc_Mumbai_p182.tif', 'ROIs0000_autumn_dfc_Mumbai_p255.tif',
               'ROIs0000_spring_dfc_BlackForest_p873.tif', 'ROIs0000_winter_dfc_MexicoCity_p440.tif',
               'ROIs0000_winter_dfc_KippaRing_p268.tif', 'ROIs0000_autumn_dfc_CapeTown_p63.tif',
               'ROIs0000_autumn_dfc_Mumbai_p256.tif', 'ROIs0000_autumn_dfc_Mumbai_p444.tif']
    args.score = True
    if not os.path.isdir(args.preview_dir):
        os.makedirs(args.preview_dir)
    # back normlize image
    image = Rnormalize_S1(image)
    image /= 25
    image += 1
    # convert to 256x256 numpy arrays
    prediction = prediction.cpu().numpy()
    prediction = np.argmax(prediction, axis=1)
    if args.score:
        target = target.cpu().numpy()

    # save predictions
    gt_id = "dfc"
    for i in range(target.shape[0]):

        # n += 1
        #id = ID[i].replace("_s2_", "_" + gt_id + "_")
        id = ID[i]

        if id in id_list:
        #if (1):
            output = labels_to_dfc(prediction[i, :, :], args.no_savanna)

            output = output.astype(np.uint8)
            #output_img = Image.fromarray(output)
            #output_img.save(os.path.join(args.out_dir, id))

            # update error metrics
            if args.score:
                gt = labels_to_dfc(target[i, :, :], args.no_savanna)
                #conf_mat.add(target[i, :, :], prediction[i, :, :])

            # save preview
            if args.preview_dir is not None:

                # colorize labels
                cmap = mycmap()
                output = (output - 1) / 10
                output = cmap(output)[:, :, 0:3]
                if args.score:
                    gt = (gt - 1) / 10
                    gt = cmap(gt)[:, :, 0:3]
                display_channels = [2, 1, 0]
                brightness_factor = 3

                if args.score:
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                else:
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                if image.shape[1] > 3:
                    img = image.cpu().numpy()[i, display_channels, :, :]
                    img = np.rollaxis(img, 0, 3)
                else:
                    img = image.cpu().numpy()[i, -2:-1, :, :]
                    img = np.rollaxis(img, 0, 3)
                ax1.imshow(np.clip(img * brightness_factor, 0, 1))
                ax1.set_title("input")
                ax1.axis("off")
                ax2.imshow(output)
                ax2.set_title("prediction")
                ax2.axis("off")
                if args.score:
                    ax3.imshow(gt)
                    ax3.set_title("label")
                    ax3.axis("off")
                lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0), handles=mypatches(), ncol=2,
                                 title="DFC Classes")
                ttl = fig.suptitle(id, y=0.75)
                plt.savefig(os.path.join(args.preview_dir, id),
                            bbox_extra_artists=(lgd, ttl,), bbox_inches='tight')
                plt.close()


def visualization(prediction, target, ID, image, args): # ID-> batch['id']
    id_list = ['ROIs0000_autumn_dfc_BandarAnzali_p611.tif', 'ROIs0000_autumn_dfc_CapeTown_p1277.tif',
               'ROIs0000_autumn_dfc_Mumbai_p182.tif', 'ROIs0000_autumn_dfc_Mumbai_p255.tif',
               'ROIs0000_spring_dfc_BlackForest_p873.tif', 'ROIs0000_winter_dfc_MexicoCity_p440.tif',
               'ROIs0000_winter_dfc_KippaRing_p268.tif', 'ROIs0000_autumn_dfc_CapeTown_p63.tif',
               'ROIs0000_autumn_dfc_Mumbai_p256.tif', 'ROIs0000_autumn_dfc_Mumbai_p444.tif']
    args.score = True
    if not os.path.isdir(args.preview_dir):
        os.makedirs(args.preview_dir)
    # back normlize image
    image = Rnormalize_S2(image)
    image /= 10000
    # convert to 256x256 numpy arrays
    prediction = prediction.cpu().numpy()
    prediction = np.argmax(prediction, axis=1)
    if args.score:
        target = target.cpu().numpy()

    # save predictions
    gt_id = "dfc"
    for i in range(target.shape[0]):

        # n += 1
        #id = ID[i].replace("_s2_", "_" + gt_id + "_")
        id = ID[i]

        if id in id_list:
        #if (1):
            output = labels_to_dfc(prediction[i, :, :], args.no_savanna)

            output = output.astype(np.uint8)
            #output_img = Image.fromarray(output)
            #output_img.save(os.path.join(args.out_dir, id))

            # update error metrics
            if args.score:
                gt = labels_to_dfc(target[i, :, :], args.no_savanna)
                #conf_mat.add(target[i, :, :], prediction[i, :, :])

            # save preview
            if args.preview_dir is not None:

                # colorize labels
                cmap = mycmap()
                output = (output - 1) / 10
                output = cmap(output)[:, :, 0:3]
                if args.score:
                    gt = (gt - 1) / 10
                    gt = cmap(gt)[:, :, 0:3]
                display_channels = [2, 1, 0]
                brightness_factor = 3

                if args.score:
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                else:
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                if image.shape[1] > 3:
                    img = image.cpu().numpy()[i, display_channels, :, :]
                    img = np.rollaxis(img, 0, 3)
                else:
                    img = image.cpu().numpy()[i, -2:-1, :, :]
                    img = np.rollaxis(img, 0, 3)
                ax1.imshow(np.clip(img * brightness_factor, 0, 1))
                ax1.set_title("input")
                ax1.axis("off")
                ax2.imshow(output)
                ax2.set_title("prediction")
                ax2.axis("off")
                if args.score:
                    ax3.imshow(gt)
                    ax3.set_title("label")
                    ax3.axis("off")
                lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0), handles=mypatches(), ncol=2,
                                 title="DFC Classes")
                ttl = fig.suptitle(id, y=0.75)
                plt.savefig(os.path.join(args.preview_dir, id),
                            bbox_extra_artists=(lgd, ttl,), bbox_inches='tight')
                plt.close()


def visualizations(target, ID, image, args): # ID-> batch['id']
    args.score = True
    if not os.path.isdir(args.preview_dir):
        os.makedirs(args.preview_dir)
    # back normlize image
    image = Rnormalize_S2(image)
    image /= 10000
    if args.score:
        target = target.cpu().numpy()

    # save predictions
    for i in range(target.shape[0]):

        id = ID[i]
        if (1):
            # update error metrics
            if args.score:
                gt = labels_to_dfc(target[i, :, :], args.no_savanna)

            # save preview
            if args.preview_dir is not None:

                # colorize labels
                cmap = mycmap()
                if args.score:
                    gt = (gt - 1) / 10
                    gt = cmap(gt)[:, :, 0:3]
                display_channels = [2, 1, 0]
                brightness_factor = 3

                if args.score:
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                else:
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                if image.shape[1] > 3:
                    img = image.cpu().numpy()[i, display_channels, :, :]
                    img = np.rollaxis(img, 0, 3)
                else:
                    img = image.cpu().numpy()[i, -2:-1, :, :]
                    img = np.rollaxis(img, 0, 3)
                ax1.imshow(np.clip(img * brightness_factor, 0, 1))
                ax1.set_title("input")
                ax1.axis("off")
                if args.score:
                    ax2.imshow(gt)
                    ax2.set_title("label")
                    ax2.axis("off")
                lgd = plt.legend(loc='lower left', bbox_to_anchor=(1.1, 0), handles=mypatches(), ncol=2,
                                 title="DFC Classes")
                ttl = fig.suptitle(id, y=0.75)
                plt.savefig(os.path.join(args.preview_dir, id), bbox_extra_artists=(lgd, ttl,), bbox_inches='tight')
                plt.close()

if __name__ == '__main__':
    meter = AverageMeter()
