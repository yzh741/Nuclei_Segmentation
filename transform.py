import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import morphology
import torch
import PIL
from torch.utils.data import DataLoader
from models.r2_unet import R2U_Net
from torchvision import transforms
from options import Options
from data_folder import get_imgs_list
from data_folder import DataFolder


# 有些只需要对img做操作，不用再mask上操作  label_encoding可以留下
class GaussianNoise(object):
    def __init__(self, min_var, max_var, p):
        self.min_var = min_var
        self.max_var = max_var
        self.p = p

    def __call__(self, imgs):
        output_imgs = list(imgs)
        img = imgs[0]  # 只需要针对原图即可
        transform = A.GaussNoise(var_limit=(self.min_var, self.max_var), p=self.p)
        transformed = transform(image=img)
        output_imgs[0] = transformed['image']
        return tuple(output_imgs)


class ChannelShuffle(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, imgs):
        output_imgs = list(imgs)
        img = imgs[0]
        transform = A.ChannelShuffle(p=self.p)
        transformed = transform(image=img)
        output_imgs[0] = transformed['image']
        return tuple(output_imgs)


class HueSaturationValue(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, imgs):
        output_imgs = list(imgs)
        img = imgs[0]
        transform = A.HueSaturationValue(p=self.p)
        transformed = transform(image=img)
        output_imgs[0] = transformed['image']
        return tuple(output_imgs)


class RandomResizedCrop(object):
    def __init__(self, height, width, p):
        self.height = height
        self.width = width
        self.p = p

    def __call__(self, imgs):
        if len(imgs) == 3:
            img = imgs[0]
            mask1 = imgs[1]
            mask2 = imgs[2]
            data = {'image': 'image', 'image1': 'image', 'image2': 'image'}
            transform = A.Compose([
                A.RandomResizedCrop(height=self.height, width=self.width, p=self.p)
            ], additional_targets=data)
            transformed = transform(image=img, image1=mask1, image2=mask2)
            output_imgs = [transformed['image'], transformed['image1'], transformed['image2']]
        else:
            img = imgs[0]
            mask1 = imgs[1]
            data = {'image': 'image', 'image1': 'image'}
            transform = A.Compose([
                A.RandomResizedCrop(height=self.height, width=self.width, p=self.p)
            ], additional_targets=data)
            transformed = transform(image=img, image1=mask1)
            output_imgs = [transformed['image'], transformed['image1']]

        return tuple(output_imgs)


class CoarseDropout(object):
    def __init__(self, max_hole, min_hole, max_height, max_width, min_height, min_width, p):
        self.max_hole = max_hole
        self.min_hole = min_hole
        self.max_height = max_height
        self.max_width = max_width
        self.min_height = min_height
        self.min_width = min_width
        self.p = p

    def __call__(self, imgs):
        if len(imgs) == 3:
            img = imgs[0]
            mask1 = imgs[1]
            mask2 = imgs[2]
            data = {'image': 'image', 'image1': 'image', 'image2': 'image'}
            transform = A.Compose([
                A.CoarseDropout(max_holes=self.max_hole, min_holes=self.max_hole, max_height=self.max_height,
                                max_width=self.max_width, min_height=self.min_height, min_width=self.min_width, p=self.p)
            ], additional_targets=data)
            transformed = transform(image=img, image1=mask1, image2=mask2)
            output_imgs = [transformed['image'], transformed['image1'], transformed['image2']]
        else:
            img = imgs[0]
            mask1 = imgs[1]
            data = {'image': 'image', 'image1': 'image'}
            transform = A.Compose([A.CoarseDropout(max_holes=self.max_hole, min_holes=self.max_hole,
                                                   max_height=self.max_height, max_width=self.max_width,
                                                   min_height=self.min_height, min_width=self.min_width, p=self.p)],
                                  additional_targets=data)
            transformed = transform(image=img, image1=mask1)
            output_imgs = [transformed['image'], transformed['image1']]
        return tuple(output_imgs)


class ElasticTransform(object):
    def __init__(self, p, alpha=50, sigma=20):
        self.p = p
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, imgs):
        if len(imgs) == 3:
            img = imgs[0]
            mask1 = imgs[1]
            mask2 = imgs[2]
            data = {'image': 'image', 'image1': 'image', 'image2': 'image'}
            transform = A.Compose([
                A.ElasticTransform(sigma=self.sigma, alpha=self.alpha, p=self.p)
            ], additional_targets=data)
            transformed = transform(image=img, image1=mask1, image2=mask2)
            output_imgs = [transformed['image'], transformed['image1'], transformed['image2']]
        else:
            img = imgs[0]
            mask1 = imgs[1]
            data = {'image': 'image', 'image1': 'image'}
            transform = A.Compose([
                A.ElasticTransform(sigma=self.sigma, alpha=self.alpha, p=self.p)
            ], additional_targets=data)
            transformed = transform(image=img, image1=mask1)
            output_imgs = [transformed['image'], transformed['image1']]
        return tuple(output_imgs)


class RandomRotate(object):
    def __init__(self, limit, p):
        self.limit = limit
        self.p = p

    def __call__(self, imgs):
        if len(imgs) == 3:
            img = imgs[0]
            mask1 = imgs[1]
            mask2 = imgs[2]
            data = {'image': 'image', 'image1': 'image', 'image2': 'image'}
            transform = A.Compose([A.Rotate(limit=self.limit, p=self.p)],
                                additional_targets=data)
            transformed = transform(image=img, image1=mask1, image2=mask2)
            output_imgs = [transformed['image'], transformed['image1'], transformed['image2']]
        else:
            img = imgs[0]
            mask1 = imgs[1]
            data = {'image': 'image', 'image1': 'image'}
            transform = A.Compose([A.Rotate(limit=self.limit, p=self.p)],
                                  additional_targets=data)
            transformed = transform(image=img, image1=mask1)
            output_imgs = [transformed['image'], transformed['image1']]
        return tuple(output_imgs)


class RandomGamma(object):
    def __init__(self, min_gamma, max_gamma, p):
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.p = p

    def __call__(self, imgs):
        output_imgs = list(imgs)
        img = imgs[0]
        transform = A.HueSaturationValue(p=self.p)
        transformed = transform(image=img)
        output_imgs[0] = transformed['image']
        return tuple(output_imgs)


class LabelEncoding(object):
    """
    Encoding the label, computes boundary individually
    """

    def __init__(self, radius=1):
        self.radius = radius

    def __call__(self, imgs):
        out_imgs = list(imgs)
        label = imgs[-1]
        if not isinstance(label, np.ndarray):
            label = np.array(label)

        # ternary label: one channel (0: background, 1: inside, 2: boundary) #
        new_label = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
        new_label[label[:, :, 0] > 255 * 0.5] = 1  # inside
        boun = morphology.dilation(new_label) & (~morphology.erosion(new_label, morphology.disk(self.radius)))
        new_label[boun > 0] = 2  # boundary

        # label = Image.fromarray(new_label.astype(np.uint8))
        out_imgs[-1] = new_label
        out_imgs = [Image.fromarray(img) for img in out_imgs]
        return tuple(out_imgs)


class ToTensor(object):
    """ Convert (img, label) of type ``PIL.Image`` or ``numpy.ndarray`` to tensors.
    Converts img of type PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    Converts label of type PIL.Image or numpy.ndarray (H x W) in the range [0, 255]
    to a torch.LongTensor of shape (H x W) in the range [0, 255].
    """

    def __init__(self, index=1):
        self.index = index  # index to distinguish between images and labels

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # for i, img in enumerate(imgs):
        #     if isinstance(img, PIL.PngImagePlugin.PngImageFile):
        #         continue
        #     else:
        #         imgs[i] = Image.fromarray(img)

        if len(imgs) < self.index:
            raise ValueError('The number of images is smaller than separation index!')

        pics = []

        # process image
        for i in range(0, self.index):
            img = imgs[i]
            if isinstance(img, np.ndarray):
                # handle numpy array
                pic = torch.from_numpy(img.transpose((2, 0, 1)))
                # backward compatibility
                pics.append(pic.float().div(255))

            # handle PIL Image
            if img.mode == 'I':
                pic = torch.from_numpy(np.array(img, np.int32, copy=False))
            elif img.mode == 'I;16':
                pic = torch.from_numpy(np.array(img, np.int16, copy=False))
            else:
                pic = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if img.mode == 'YCbCr':
                nchannel = 3
            elif img.mode == 'I;16':
                nchannel = 1
            else:
                nchannel = len(img.mode)
            pic = pic.view(img.size[1], img.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            pic = pic.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(pic, torch.ByteTensor):
                pics.append(pic.float().div(255))
            else:
                pics.append(pic)

        # process labels:
        for i in range(self.index, len(imgs)):
            # process label
            label = imgs[i]
            if isinstance(label, np.ndarray):
                # handle numpy array
                label_tensor = torch.from_numpy(label)
                # backward compatibility
                pics.append(label_tensor.long())

            # handle PIL Image
            if label.mode == 'I':
                label_tensor = torch.from_numpy(np.array(label, np.int32, copy=False))
            elif label.mode == 'I;16':
                label_tensor = torch.from_numpy(np.array(label, np.int16, copy=False))
            else:
                label_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(label.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if label.mode == 'YCbCr':
                nchannel = 3
            elif label.mode == 'I;16':
                nchannel = 1
            else:
                nchannel = len(label.mode)
            label_tensor = label_tensor.view(label.size[1], label.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            label_tensor = label_tensor.transpose(0, 1).transpose(0, 2).contiguous()
            # label_tensor = label_tensor.view(label.size[1], label.size[0])
            pics.append(label_tensor.long())

        return tuple(pics)


class Normalize(object):
    """ Normalize an tensor image with mean and standard deviation.
    Given mean and std, will normalize each channel of the torch.*Tensor,
     i.e. channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    ** only normalize the first image, keep the target image unchanged
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensors):
        """
        Args:
            tensors (Tensor): Tensor images of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensors = list(tensors)
        for t, m, s in zip(tensors[0], self.mean, self.std):
            t.sub_(m).div_(s)
        return tuple(tensors)


selector = {
    'GuassianNoise': lambda x: GaussianNoise(x[0], x[1], x[2]),
    'ChannelShuffle': lambda x: ChannelShuffle(x),
    'HueSaturationValue': lambda x: HueSaturationValue(x),
    'RandomResizedCrop': lambda x: RandomResizedCrop(x[0], x[1], x[2]),
    'CoarseDropout': lambda x: CoarseDropout(*x),
    'ElasticTransform': lambda x: ElasticTransform(x),
    'RandomRotate': lambda x: RandomRotate(x[0], x[1]),
    'RandomGamma': lambda x: RandomGamma(x[0], x[1], x[2]),
    'label_encoding': lambda x: LabelEncoding(x),
    'to_tensor': lambda x: ToTensor(x),
    'normalize': lambda x: Normalize(x[0], x[1])
}


def get_transforms(param_dict):
    """ data transforms for train, validation or test """
    t_list = []
    for k, v in param_dict.items():
        t_list.append(selector[k](v))
    return Compose(t_list)


class Compose(object):
    """ Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        for t in self.transforms:
            imgs = t(imgs)
        return imgs



