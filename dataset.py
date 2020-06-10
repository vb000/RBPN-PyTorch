import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import cv2
from skimage import img_as_float
from random import randrange
import os.path
import glob
import re
import sys

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_img(filepath, nFrames, scale, other_dataset):
    seq = [i for i in range(1, nFrames)]
    #random.shuffle(seq) #if random sequence
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('L'),scale)
        input=target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]
        neigbor_hd = []

        for i in seq:
            # index = int(filepath[char_len-7:char_len-4])-i
            index = int(re.search('frame(.*).jpg', filepath).group(1))-i
            file_path = re.search('(.*)frame', filepath).group(1)
            file_name = file_path + 'frame{0}'.format(index) + '.jpg'
            if os.path.exists(file_name):
                temp = modcrop(Image.open(file_name).convert('L'),scale)
                neigbor_hd.append(temp)
                temp = temp.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame is not exist')
                temp = input
                neigbor.append(temp)
                neigbor_hd.append(target)
    else:
        target = modcrop(Image.open(join(filepath,'im'+str(nFrames)+'.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = [modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC) for j in reversed(seq)]
    
    return target, input, neigbor, neigbor_hd

def load_img_future(filepath, nFrames, scale, other_dataset):
    tt = int(nFrames/2)
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('L'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        
        char_len = len(filepath)
        neigbor=[]
        neigbor_hd = []
        if nFrames%2 == 0:
            seq = [x for x in range(-tt,tt) if x!=0] # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
        else:
            seq = [x for x in range(-tt,tt+1) if x!=0]
        #random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(re.search('frame(.*).jpg', filepath).group(1))+i
            # index1 = int(filepath[-7:char_len-4])+i
            file_path1 = re.search('(.*)frame', filepath).group(1)
            file_name1=file_path1+'frame{0}'.format(index1)+'.jpg'
            if os.path.exists(file_name1):

                temp = modcrop(Image.open(file_name1).convert('L'), scale)
                neigbor_hd.append(temp)
                temp = temp.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame- is not exist')
                temp=input
                neigbor.append(temp)
                neigbor_hd.append(target)
            
    else:
        target = modcrop(Image.open(join(filepath,'im4.png')).convert('RGB'),scale)
        input = target.resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC)
        neigbor = []
        seq = [x for x in range(4-tt,5+tt) if x!=4]
        #random.shuffle(seq) #if random sequence
        for j in seq:
            neigbor.append(modcrop(Image.open(filepath+'/im'+str(j)+'.png').convert('RGB'), scale).resize((int(target.size[0]/scale),int(target.size[1]/scale)), Image.BICUBIC))
    return target, input, neigbor, neigbor_hd

def get_flow(im1, im2):
    im1 = np.array(im1)
    im2 = np.array(im2)
    flow = cv2.calcOpticalFlowFarneback(im1, im2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow

def rescale_flow(x,max_range,min_range):
    max_val = np.max(x)
    min_val = np.min(x)
    return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range

def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih%modulo);
    iw = iw - (iw%modulo);
    img = img.crop((0, 0, ih, iw))
    return img

def get_patch(img_in, img_tar, img_nn, img_nn_hd, patch_size, scale, nFrames, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale
    try:
        if ix == -1:
            ix = random.randrange(0, iw - ip + 1)
        if iy == -1:
            iy = random.randrange(0, ih - ip + 1)
    except:
        print(ip)
        print(iw)
        print(ih)
        blah.blah

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy,ix,iy + ip, ix + ip))#[:, iy:iy + ip, ix:ix + ip]
    img_tar = img_tar.crop((ty,tx,ty + tp, tx + tp))#[:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.crop((iy,ix,iy + ip, ix + ip)) for j in img_nn] #[:, iy:iy + ip, ix:ix + ip]
    img_nn_hd = [j.crop((ty,tx,ty + tp, tx + tp)) for j in img_nn_hd] #[:, ty:ty + tp, tx:tx + tp]
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_nn, img_nn_hd, info_patch

def augment(img_in, img_tar, img_nn, img_nn_hd, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_nn = [ImageOps.flip(j) for j in img_nn]
        img_nn_hd = [ImageOps.flip(j) for j in img_nn_hd]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            img_nn_hd = [ImageOps.mirror(j) for j in img_nn_hd]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_nn = [j.rotate(180) for j in img_nn]
            img_nn_hd = [j.rotate(180) for j in img_nn_hd]
            info_aug['trans'] = True

    return img_in, img_tar, img_nn, img_nn_hd, info_aug
    
def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir,nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size, future_frame, transform=None, epoch_size=None):
        super(DatasetFromFolder, self).__init__()
        # alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        # self.image_filenames = [join(image_dir,x) for x in alist]
        # print(image_dir+"/**/*.jpg")
        self.image_filenames = glob.glob(image_dir + '/**/*.jpg', recursive=True)
        # print(self.image_filenames)
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.future_frame = future_frame
        self.epoch_size = epoch_size
        if epoch_size is None:
            self.epoch_size = len(self.image_filenames)

    def __getitem__(self, index):
        if self.epoch_size != len(self.image_filenames):
            # Ignore, choose random
            index = np.random.randint(len(self.image_filenames))

        if self.future_frame:
            target, input, neigbor, neigbor_hd = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            target, input, neigbor, neigbor_hd = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        try:
            if self.patch_size != 0:
                input, target, neigbor, neigbor_hd, _ = get_patch(input,target,neigbor,neigbor_hd, self.patch_size, self.upscale_factor, self.nFrames)
        except:
            print("Error in file: " + self.image_filenames[index])
            sys.exit(-1)

        if self.data_augmentation:
            input, target, neigbor, neigbor_hd, _ = augment(input, target, neigbor, neigbor_hd)
            
        flow = [get_flow(input,j) for j in neigbor]
            
        bicubic = rescale_img(input, self.upscale_factor)
        
        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]
            neigbor_hd = [self.transform(j) for j in neigbor_hd]
            flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]

        return input, target, neigbor, neigbor_hd, flow, bicubic

    def __len__(self):
        return self.epoch_size

class DatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None, epoch_size=None):
        super(DatasetFromFolderTest, self).__init__()
        # alist = [line.rstrip() for line in open(join(image_dir,file_list))]
        # self.image_filenames = [join(image_dir,x) for x in alist]
        self.image_filenames = glob.glob(image_dir + '/**/*.jpg', recursive=True)
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.other_dataset = other_dataset
        self.future_frame = future_frame
        self.epoch_size = epoch_size
        if epoch_size is None:
            self.epoch_size = len(self.image_filenames)

    def __getitem__(self, index):
        if self.epoch_size != len(self.image_filenames):
            # Ignore, choose random
            index = np.random.randint(len(self.image_filenames))

        if self.future_frame:
            target, input, neigbor, neigbor_hd = load_img_future(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
        else:
            target, input, neigbor, neigbor_hd = load_img(self.image_filenames[index], self.nFrames, self.upscale_factor, self.other_dataset)
            
        flow = [get_flow(input,j) for j in neigbor]

        bicubic = rescale_img(input, self.upscale_factor)
        
        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]
            neigbor_hd = [self.transform(j) for j in neigbor_hd]
            flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]
            
        return input, target, neigbor, neigbor_hd, flow, bicubic, self.image_filenames[index]
      
    def __len__(self):
        return self.epoch_size
