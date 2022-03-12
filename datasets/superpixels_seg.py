import os
import glob
import rasterio
import numpy as np
from tqdm import tqdm

import torch.utils.data as data
from skimage.segmentation import felzenszwalb, slic, mark_boundaries

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

# indices of sentinel-2 high-/medium-/low-resolution bands
S2_BANDS_HR = [2, 3, 4]
S2_BANDS_MR = [5, 6, 7, 9, 12, 13]
S2_BANDS_LR = [1, 10, 11]


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
    s2 = np.clip(s2, 0, 10000)
    return s2


# util function for reading s1 data
def load_s1(path):
    with rasterio.open(path) as data:
        s1 = data.read()
    s1 = s1.astype(np.float32)
    s1 = np.nan_to_num(s1)
    s1 = np.clip(s1, -25, 0)
    # normaliza 0~1
    s1 /= 25
    s1 += 1
    s1 = s1.astype(np.float32)
    return s1


# this function for classification and most important is for weak supervised
def load_sample(sample, use_s1, use_s2hr, use_s2mr, use_s2lr, superpixel=True,
                no_savanna=False, igbp=True, unlabeled=False, n_segments=100, sigma=2):

    use_s2 = use_s2hr or use_s2mr or use_s2lr

    # load s2 data
    if use_s2:
        img = load_s2(sample["s2"], use_s2hr, use_s2mr, use_s2lr)
        s2 = normalization(img)  # normaliza 0~1
        s2 = s2.astype(np.float32)
        #s2 = s2.swapaxes(2, 0) #这个错了,全错了
        s2 = np.rollaxis(s2, 0, 3)
        segments = felzenszwalb(s2, scale=200, sigma=0.50, min_size=30)
        #segments = felzenszwalb(s2, scale=50, sigma=0.80, min_size=30)
        segments = segments + 1 #为了和slic保持一致
        #segments = slic(s2, n_segments=1000, sigma=1, start_label=1, multichannel=True)
        print(segments.max())
        print(sample["s2"].replace("tif", "npy").replace("s2_", "se_"))
        #print(os.path.split(sample["s2"].replace("tif", "npy").replace("s2_", "se_"))[0])
        if not os.path.isdir(os.path.dirname(sample["s2"].replace("tif", "npy").replace("s2_", "se_"))):
            os.makedirs(os.path.dirname(sample["s2"].replace("tif", "npy").replace("s2_", "se_")))
        np.save(sample["s2"].replace("tif", "npy").replace("s2_", "se_"), segments)

    # segmentate the image to superpixels
    if superpixel:
        segments = None
    else:
        segments = None

    # load label
    if unlabeled:
        return {'image': img, 'segments': segments, 'id': sample["id"]}
    else:
        return {'image': img, 'segments': segments, 'id': sample["id"]}



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
                 train_index=None):
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
        self.train_index = train_index
        assert subset in ["val", "train", "test"]
        self.no_savanna = no_savanna
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
        else:
            path = os.path.join(path, "ROIs0000_test", "s2_0")
            sample_dirs = []

        self.samples = []
        for folder in sample_dirs:
            s2_locations = glob.glob(os.path.join(path, f"{folder}/*.tif"), recursive=True)
            for s2_loc in tqdm(s2_locations, desc="[Load]"):
                self.samples.append({"s2": s2_loc, "id": os.path.basename(s2_loc)})
        # sort list of samples
        #self.samples = sorted(self.samples, key=lambda i: i['id'])
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
        return load_sample(sample, self.use_s1, self.use_s2hr, self.use_s2mr,
                           self.use_s2lr, no_savanna=self.no_savanna, igbp=False)

    def __len__(self):
        """Get number of samples in the dataset"""
        return len(self.samples)


if __name__ == "__main__":
    print("\n\nDFC2020 test")
    data_dir = '/workplace/DFC2020'
    ds = DFC2020(data_dir, subset="train", use_s1=False, use_s2hr=True, use_s2mr=False, use_s2lr=False, no_savanna=True, train_index='../utils/train_100.npy')
    for i in range(len(ds)):
        s = ds.__getitem__(i)
        print("id:", s["id"], "\n", "input shape:", s["image"].shape)
