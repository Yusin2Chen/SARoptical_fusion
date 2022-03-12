import os
import glob
import rasterio
import numpy as np
from tqdm import tqdm
import torch.utils.data as data

# standar
S2_MEAN = np.array([1353.3418, 1265.4015, 1269.009, 1976.1317])
S2_STD  = np.array([242.07303, 290.84450, 402.9476, 516.77480])

def normalize_S2(imgs):
    for i in range(4):
        imgs[i,:,:] = (imgs[i, :,:] - S2_MEAN[i]) / S2_STD[i]
    return imgs

S1_MEAN = np.array([-9.020017, -15.73008])
S1_STD  = np.array([3.5793820, 3.671725])

def normalize_S1(imgs):
    for i in range(2):
        imgs[i,:,:] = (imgs[i, :,:] - S1_MEAN[i]) / S1_STD[i]
    return imgs

def MEAN_HV(img):
    return (img[0,:] + img[1,:]) / 2

def NDBI(img):
    #(B11 - B8) / (B11 + B8)
    return (img[11,:] - img[7,:]) / (img[11,:] + img[7,:])

def NDVI(img):
    #(B8 - B4) / (B8 + B4)
    return (img[7,:] - img[3,:]) / (img[7,:] + img[3,:])

def DBSI(img):
    #(B11 - B3)/(B11 + B3) - NDVI(img)
    return (img[11, :] - img[2, :]) / (img[11, :] + img[2, :]) - NDVI(img)

def NDWI(img):
    #(B03 - B08) / (B03 + B08)
    return (img[2, :] - img[7, :]) / (img[2, :] + img[7, :])



# data augmenttaion
class RandomCrop(object):
    """给定图片，随机裁剪其任意一个和给定大小一样大的区域.

    Args:
        output_size (tuple or int): 期望裁剪的图片大小。如果是 int，将得到一个正方形大小的图片.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample, unlabeld=True, superpixel=True):

        if unlabeld:
            image, id = sample['image'], sample['id']
            lc = None
        else:
            image, lc, id = sample['image'], sample['label'], sample['id']

        _, h, w = image.shape
        new_h, new_w = self.output_size
        # 随机选择裁剪区域的左上角，即起点，(left, top)，范围是由原始大小-输出大小
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top: top + new_h, left: left + new_w]

        if superpixel:
            segments = sample["segments"]
            segments = segments[top: top + new_h, left: left + new_w]
        else:
            segments = None
        # index
        index = sample["index"]
        index = index[:, top: top + new_h, left: left + new_w]

        # load label
        if unlabeld:
            return {'image': image, 'segments': segments, 'index': index, 'id': id}
        else:
            lc = lc[top: top + new_h, left: left + new_w]
            return {'image': image, 'segments': segments, 'index': index, 'label': lc, 'id': id}

# mapping from igbp to dfc2020 classes
DFC2020_CLASSES = [
    0,  # class 0 unused in both schemes
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    3,  # --> will be masked if no_savanna == True
    3,  # --> will be masked if no_savanna == True
    4,
    5,
    6,  # 12 --> 6
    7,  # 13 --> 7
    6,  # 14 --> 6
    8,
    9,
    10
    ]

# indices of sentinel-2 high-/medium-/low-resolution bands
S2_BANDS_HR = [2, 3, 4, 8]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [1, 10, 11]


# util function for reading s2 data
def load_s2(path, use_hr, use_mr, use_lr):
    bands_selected = []
    #bands_selected1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    if use_hr:
        bands_selected = bands_selected + S2_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + S2_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + S2_BANDS_LR
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as data:
        s2 = data.read(bands_selected)
    s2 = s2.astype(np.float32)
    s2 = np.clip(s2, 0, 10000)
    s2 = normalize_S2(s2)
    #s2 /= 10000
    return s2


# util function for reading s2 data
def load_index(path1, path2):

    with rasterio.open(path1) as data:
        s1 = data.read([1, 2])
    s1 = s1.astype(np.float32)
    s1 = np.clip(s1, -25, 0)
    hv = MEAN_HV(s1)

    bands_selected = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    with rasterio.open(path2) as data:
        s2 = data.read(bands_selected)
    s2 = s2.astype(np.float32)
    s2 = np.clip(s2, 0, 10000)
    #bi = NDBI(s2)
    vi = NDVI(s2)
    si = DBSI(s2)
    wi = NDWI(s2)
    #index = np.array([hv, bi, vi, si, wi])
    index = np.array([hv, vi, si, wi])
    return index


# util function for reading s1 data
def load_s1(path):
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    s1 = normalize_S1(s1)
    #s1 /= 25
    #s1 += 1
    return s1


# util function for reading lc data
def load_lc(path, no_savanna=False, igbp=True):

    # load labels
    with rasterio.open(path) as data:
        lc = data.read(1)

    # convert IGBP to dfc2020 classes
    if igbp:
        lc = np.take(DFC2020_CLASSES, lc)
    else:
        lc = lc.astype(np.int64)

    # adjust class scheme to ignore class savanna
    if no_savanna:
        lc[lc == 3] = 0
        lc[lc > 3] -= 1

    # convert to zero-based labels and set ignore mask
    lc -= 1
    lc[lc == -1] = 255
    return lc

# this function for classification and most important is for weak supervised
def load_sample(sample, use_s1, use_s2hr, use_s2mr, use_s2lr,
                no_savanna=False, igbp=True, unlabeled=False, superpixel=True):

    use_s2 = use_s2hr or use_s2mr or use_s2lr

    # load s2 data
    if use_s2:
        img = load_s2(sample["s2"], use_s2hr, use_s2mr, use_s2lr)
    else:
        img = None

    # load s1 data
    if use_s1:
        if use_s2:
            img = np.concatenate((img, load_s1(sample["s1"])), axis=0)
        else:
            img = load_s1(sample["s1"])

    # segmentate the image to superpixels
    if superpixel:
        segments = np.load(sample["se"])
    else:
        segments = None

    # load index image
    index = load_index(sample["s1"], sample["s2"])

    # load label
    if unlabeled:
        return {'image': img, 'segments': segments, 'index': index, 'id': sample["id"]}
    else:
        lc = load_lc(sample["lc"], no_savanna=no_savanna, igbp=igbp)
        return {'image': img, 'segments': segments, 'index': index, 'label': lc, 'id': sample["id"]}


# calculate number of input channels
def get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr):
    n_inputs = 0
    if use_s2hr:
        n_inputs += len(S2_BANDS_HR)
    if use_s2mr:
        n_inputs += len(S2_BANDS_MR)
    if use_s2lr:
        n_inputs += len(S2_BANDS_LR)
    if use_s1:
        n_inputs += 2
    return n_inputs


# select channels for preview images
def get_display_channels(use_s2hr, use_s2mr, use_s2lr):
    if use_s2hr and use_s2lr:
        display_channels = [3, 2, 1]
        brightness_factor = 3
    elif use_s2hr:
        display_channels = [2, 1, 0]
        brightness_factor = 3
    elif not (use_s2hr or use_s2mr or use_s2lr):
        display_channels = 0
        brightness_factor = 1
    else:
        display_channels = 0
        brightness_factor = 3
    return (display_channels, brightness_factor)


class DFC2020(data.Dataset):
    """PyTorch dataset class for the DFC2020 dataset"""

    def __init__(self,
                 path,
                 subset="val",
                 no_savanna=False,
                 use_s2hr=False,
                 use_s2mr=False,
                 use_s2lr=False,
                 use_s1=False,
                 unlabeled=True,
                 transform=False,
                 train_index=None,
                 crop_size=32):
        """Initialize the dataset"""

        # inizialize
        super(DFC2020, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1
        assert subset in ["val", "train", "test"]
        self.no_savanna = no_savanna
        self.unlabeled = unlabeled
        self.train_index = train_index
        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(use_s2hr, use_s2mr, use_s2lr)

        # provide number of classes
        if no_savanna:
            self.n_classes = max(DFC2020_CLASSES) - 1
        else:
            self.n_classes = max(DFC2020_CLASSES)

        # define transform
        if transform:
            self.transform = RandomCrop(crop_size)
        else:
            self.transform = None
        # make sure parent dir exists
        assert os.path.exists(path)

        # build list of sample paths
        if subset == "train":
            train_list = []
            for seasonfolder in ['ROIs0000_autumn', 'ROIs0000_spring',
                                 'ROIs0000_winter', 'ROIs0000_summer']:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder))]
            train_list = [x for x in train_list if "s2_" in x]
            sample_dirs = train_list
            # path = os.path.join(path, "ROIs0000_validation", "s2_validation")
        else:
            path = os.path.join(path, "ROIs0000_test", "s2_0")
            sample_dirs = []

        self.samples = []
        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"), recursive=True)
            for s2_loc in tqdm(s2_locations, desc="[Load]"):
                s1_loc = s2_loc.replace("_s2_", "_s1_").replace("s2_", "s1_")
                se_loc = s2_loc.replace("tif", "npy").replace("s2_", "se_").replace("_s2_", "_se_")
                lc_loc = s2_loc.replace("_s2_", "_dfc_").replace("s2_", "dfc_")
                self.samples.append({"lc": lc_loc, "s1": s1_loc, "s2": s2_loc, "se": se_loc, "id": os.path.basename(lc_loc)})

        # sort list of samples
        if self.train_index:
            Tindex = np.load(self.train_index)
            self.samples = [self.samples[i] for i in Tindex]

        print("loaded", len(self.samples),
              "samples from the dfc2020 subset", subset)

    def __getitem__(self, index):
        """Get a single example from the dataset"""

        # get and load sample from index file
        sample = self.samples[index]
        data_sample = load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                                  self.use_s2lr, no_savanna=self.no_savanna,
                                  igbp=False, unlabeled=self.unlabeled)
        if self.transform:
            return self.transform(data_sample)
        else:
            return data_sample

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


if __name__ == "__main__":
    print("\n\nDFC2020 test")
    data_dir = '../CMC/DFC2020'
    ds = DFC2020(data_dir, subset="train", use_s1=False, use_s2hr=True, use_s2mr=False, no_savanna=True)
    s, index = ds.__getitem__(0)
    print("id:", s["id"], "\n",
          "input shape:", s["image"].shape, "\n",
          "imput label:", s["lc"])
