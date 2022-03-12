import os
import glob
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import combinations
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

L8_MEAN = np.array([0.13946152, 0.12857966, 0.12797806, 0.23040992])
L8_STD  = np.array([0.01898952, 0.02437881, 0.03323532, 0.04915179])

def normalize_L8(imgs):
    for i in range(4):
        imgs[i,:,:] = (imgs[i, :,:] - L8_MEAN[i]) / L8_STD[i]
    return imgs

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
            segments_small = sample["segments_small"]
            segments_small = segments_small[top: top + new_h, left: left + new_w]
        else:
            segments = None
            segments_small = None

        # load label
        if unlabeld:
            return {'image': image, 'segments': segments, 'segments_small':segments_small, 'id': id}
        else:
            lc = lc[top: top + new_h, left: left + new_w]
            return {'image': image, 'segments': segments, 'segments_small':segments_small, 'label': lc, 'id': id}


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

L8_BANDS_HR = [2, 3, 4, 5]
L8_BANDS_MR = [5, 6, 7, 9, 12, 13]
L8_BANDS_LR = [1, 10, 11]

# util function for reading s2 data
def load_s2(path, use_hr, use_mr, use_lr):
    bands_selected = []
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
    s2 = s2[:,0:512,0:512]
    s2 = np.clip(s2, 0, 10000)
    s2 = normalize_S2(s2)
    return s2

def load_l8(path, use_hr, use_mr, use_lr):
    bands_selected = []
    if use_hr:
        bands_selected = bands_selected + L8_BANDS_HR
    if use_mr:
        bands_selected = bands_selected + L8_BANDS_MR
    if use_lr:
        bands_selected = bands_selected + L8_BANDS_LR
    bands_selected = sorted(bands_selected)
    with rasterio.open(path) as data:
        l8 = data.read(bands_selected)
    l8 = l8.astype(np.float32)
    l8 = np.clip(l8, 0, 1)
    l8 = normalize_L8(l8)
    return l8

# util function for reading s1 data
def load_s1(path):
    with rasterio.open(path) as data:
        s1 = data.read([1, 2])
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    s1 = normalize_S1(s1)
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
            img = np.concatenate((img, load_s2(sample["s1"], use_s2hr, use_s2mr, use_s2lr)), axis=0)
        else:
            img = load_s2(sample["s1"], use_s2hr, use_s2mr, use_s2lr)

    # segmentate the image to superpixels
    if superpixel:
        segments = np.load(sample["seb"])
        segments = segments[0:512, 0:512]
        segments_small = np.load(sample["ses"])
        segments_small = segments_small[0:512, 0:512]
    else:
        segments = None
        segments_small = None


    # load label
    if unlabeled:
        return {'image': img, 'segments': segments, 'segments_small': segments_small, 'id': sample["id"]}
    else:
        lc = load_lc(sample["lc"], no_savanna=no_savanna, igbp=igbp)
        return {'image': img, 'segments': segments, 'segments_small': segments_small, 'label': lc, 'id': sample["id"]}


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


class OSCD_S2(data.Dataset):
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
                 train_index = None,
                 crop_size = 32):
        """Initialize the dataset"""

        # inizialize
        super(OSCD_S2, self).__init__()

        # make sure parameters are okay
        if not (use_s2hr or use_s2mr or use_s2lr or use_s1):
            raise ValueError("No input specified, set at least one of "
                             + "use_[s2hr, s2mr, s2lr, s1] to True!")
        self.use_s2hr = use_s2hr
        self.use_s2mr = use_s2mr
        self.use_s2lr = use_s2lr
        self.use_s1 = use_s1
        self.unlabeled = unlabeled
        assert subset in ["val", "train", "test"]
        self.no_savanna = no_savanna
        self.train_index = train_index
        # provide number of input channels
        self.n_inputs = get_ninputs(use_s1, use_s2hr, use_s2mr, use_s2lr)

        # provide index of channel(s) suitable for previewing the input
        self.display_channels, self.brightness_factor = get_display_channels(
                                                            use_s2hr,
                                                            use_s2mr,
                                                            use_s2lr)
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
            #for seasonfolder in ['abudhabi', 'aguasclaras', 'beihai', 'beirut', 'bercy', 'bordeaux', 'brasilia',
            #                     'chongqing', 'cupertino', 'dubai', 'hongkong', 'lasvegas', 'milano', 'montpellier',
            #                     'mumbai', 'nantes', 'norcia', 'paris', 'pisa', 'rennes', 'rio', 'saclaye',
            #                     'saclayw', 'valencia']:
            #for seasonfolder in ['abudhabi', 'cupertino', 'L15-0387E-1276N', 'L15-0586E-1127N', 'L15-1014E-1375N',
            #                     'L15-1203E-1203N', 'L15-1298E-1322N', 'L15-1615E-1205N', 'L15-1691E-1211N',
            #                     'montpellier', 'saclaye',
            #                     'aguasclaras', 'dubai', 'L15-0434E-1218N', 'L15-0595E-1278N', 'L15-1015E-1062N',
            #                     'L15-1204E-1202N', 'L15-1335E-1166N', 'L15-1615E-1206N', 'L15-1703E-1219N', 'mumbai',
            #                     'saclayw',
            #                     'beihai', 'hongkong', 'L15-0457E-1135N', 'L15-0614E-0946N', 'L15-1025E-1366N',
            #                     'L15-1204E-1204N', 'L15-1389E-1284N', 'L15-1617E-1207N', 'L15-1709E-1112N', 'nantes',
            #                     'valencia',
            #                     'beirut', 'L15-0331E-1257N', 'L15-0487E-1246N', 'L15-0632E-0892N', 'L15-1049E-1370N',
            #                     'L15-1209E-1113N', 'L15-1438E-1134N', 'L15-1669E-1153N', 'L15-1716E-1211N', 'norcia',
            #                     'bercy', 'L15-0357E-1223N', 'L15-0506E-1204N', 'L15-0683E-1006N', 'L15-1138E-1216N',
            #                     'L15-1210E-1025N', 'L15-1439E-1134N', 'L15-1669E-1159N', 'L15-1748E-1247N', 'paris',
            #                     'bordeaux', 'L15-0358E-1220N', 'L15-0544E-1228N', 'L15-0760E-0887N', 'L15-1172E-1306N',
            #                    'L15-1276E-1107N', 'L15-1479E-1101N', 'L15-1669E-1160N', 'L15-1848E-0793N',
            #                     'brasilia', 'L15-0361E-1300N', 'L15-0566E-1185N', 'L15-0924E-1108N', 'L15-1185E-0935N',
            #                    'L15-1289E-1169N', 'L15-1481E-1119N', 'L15-1672E-1207N', 'lasvegas', 'rennes',
            #                    'chongqing', 'L15-0368E-1245N', 'L15-0577E-1243N', 'L15-0977E-1187N',
            #                     'L15-1200E-0847N', 'L15-1296E-1198N', 'L15-1538E-1163N', 'L15-1690E-1211N', 'milano',
            #                     'rio']:
            for seasonfolder in ['abudhabi',     'chongqing',       'L15-0361E-1300',  'L15-0487E-1246',  'L15-0586E-1127',  'L15-0760E-0887',  'L15-1049E-1370',  'L15-1204E-1204',  'L15-1289E-1169',  'L15-1479E-1101',  'L15-1630E-0988',  'L15-1690E-1211',  'L15-1848E-0793',  'paris',
                'aguasclaras',  'cupertino',       'L15-0368E-1245',  'L15-0506E-1204',  'L15-0595E-1278',  'L15-0924E-1108',  'L15-1129E-0819',  'L15-1209E-1113',  'L15-1296E-1198',  'L15-1481E-1119',  'L15-1666E-1189',  'L15-1691E-1211',  'lasvegas',        'pisa',
                'beihai',       'dubai',           'L15-0369E-1244',  'L15-0509E-1108',  'L15-0614E-0946',  'L15-0977E-1187',  'L15-1138E-1216',  'L15-1210E-1025',  'L15-1298E-1322',  'L15-1538E-1163',  'L15-1669E-1153',  'L15-1703E-1219',  'milano',          'rennes',
                'beirut',       'hongkong',        'L15-0387E-1276',  'L15-0544E-1228',  'L15-0632E-0892',  'L15-1014E-1375',  'L15-1172E-1306',  'L15-1213E-1238',  'L15-1389E-1284',  'L15-1546E-1154',  'L15-1669E-1160',  'L15-1709E-1112',  'montpellier',     'rio',
                'bercy',        'L15-0331E-1257',  'L15-0391E-1219',  'L15-0566E-1185',  'L15-0683E-1006',  'L15-1015E-1062',  'L15-1185E-0935',  'L15-1249E-1167',  'L15-1438E-1134',  'L15-1615E-1205',  'L15-1670E-1159',  'L15-1716E-1211',  'mumbai',          'saclaye',
                'bordeaux',     'L15-0357E-1223',  'L15-0434E-1218',  'L15-0571E-1302',  'L15-0697E-0874',  'L15-1025E-1366',  'L15-1203E-1203',  'L15-1276E-1107',  'L15-1438E-1227',  'L15-1615E-1206',  'L15-1672E-1207',  'L15-1748E-1247',  'nantes',          'saclayw',
                'brasilia',     'L15-0358E-1220',  'L15-0457E-1135',  'L15-0577E-1243',  'L15-0744E-0927',  'L15-1031E-1300',  'L15-1204E-1202',  'L15-1281E-1035',  'L15-1439E-1134',  'L15-1617E-1207',  'L15-1690E-1210',  'L15-1749E-1266',  'norcia',          'valencia']:
                train_list += [os.path.join(seasonfolder, x) for x in
                               os.listdir(os.path.join(path, seasonfolder)) if ("s2_" in x and "s2_0s" not in x) ]
            train_list = [os.path.join(x, y) for x in train_list for y in os.listdir(os.path.join(path, x))]
            sample_dirs = train_list
            #path = os.path.join(path, "ROIs0000_validation", "s2_validation")
        else:
            path = os.path.join(path, "ROIs0000_test", "s2_0")
            sample_dirs = []

        self.samples = []
        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"), recursive=True)
            s2_locations = sorted(s2_locations, key=lambda x: int(os.path.basename(x)[0:6]))
            # 只取一对
            if len(s2_locations)>1:
                #s2_locations = s2_locations[0:2]
                comb_list = list(combinations(s2_locations, 2))
                # 下面这句用来筛出跨时过长的 <10
                comb_list = [comb_list[i] for i in range(len(comb_list))
                             if ((int(os.path.basename(comb_list[i][0])[0:6]) - int(os.path.basename(comb_list[i][1])[0:6])) <= -1) and ((int(os.path.basename(comb_list[i][0])[0:6]) - int(os.path.basename(comb_list[i][1])[0:6])) >= -6)]
                for (s1_loc, s2_loc) in tqdm(comb_list, desc="[Load]"):
                    seb_loc = s2_loc.replace("tif", "npy").replace("s2_", "seb_")
                    ses_loc = s2_loc.replace("tif", "npy").replace("s2_", "ses_")
                    self.samples.append({"s1": s1_loc, "s2": s2_loc, "seb": seb_loc, "ses": ses_loc, "id": os.path.basename(s2_loc)})


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


# DEBUG usage examples (expects sen12ms in /root/data
#       dfc2020 data in /root/val and unlabeled data in /root/test)
if __name__ == "__main__":

    print("\n\nOSCD_S2 validation")
    data_dir = "/workplace/S2BYOL/OSCD"
    ds = OSCD_S2(data_dir, subset="train", use_s1=True, use_s2hr=True,
                 use_s2mr=True, no_savanna=True)
    s = ds.__getitem__(0)
    print("id:", s["id"], "\n",
          "input shape:", s["image"].shape, "\n")


