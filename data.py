from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, ToTensor

from dataset import DatasetFromFolderTest, DatasetFromFolder

def transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame, epoch_size=None):
    return DatasetFromFolder(data_dir,nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,future_frame,
                             transform=transform(), epoch_size=epoch_size)


def get_eval_set(data_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame, epoch_size=None):
    return DatasetFromFolder(data_dir,nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,future_frame,
                             transform=transform(), epoch_size=epoch_size)

def get_test_set(data_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, epoch_size=None):
    return DatasetFromFolderTest(data_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=transform(), epoch_size=epoch_size)

