from __future__ import print_function
import argparse

import os
import torch
from torch.autograd import Variable
from rbpn import Net as RBPN
from dataset import load_img, get_flow, rescale_img
from data import transform
import numpy as np

import matplotlib.pyplot as plt

import time
import cv2
import math

import sys

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./Vid4')
parser.add_argument('--future_frame', type=bool, default=False, help="use future frame")
parser.add_argument('--nFrames', type=int, default=2)
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--output', default='Results/', help='Location to images and plots')
parser.add_argument('--model', default='weights/4x_tiktokRBPNF7_epoch_250.pth', help='sr pretrained base model')
parser.add_argument('--reset_hd', type=int, default=0, help='how often to reset HD frame')

opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Building model ', opt.model_type)
if opt.model_type == 'RBPN':
    model = RBPN(num_channels=1, base_filter=256, feat = 64, num_stages=3, n_resblock=5, nFrames=opt.nFrames, scale_factor=opt.upscale_factor)

if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])

def processFrame(imgFile, hdOverride = None):

    # Set Model to Eval
    model.eval()

    # Load image and optical flow
    target, input, neigbor, neigbor_hd = load_img(imgFile, opt.nFrames, opt.upscale_factor, True)
    flow = [get_flow(input,j) for j in neigbor]
    bicubic = rescale_img(input, opt.upscale_factor)

    # Convert to Torch
    target = transform()(target)
    input = transform()(input)
    bicubic = transform()(bicubic)
    neigbor = [transform()(j) for j in neigbor]
    neigbor_hd = [transform()(j) for j in neigbor_hd]
    flow = [torch.from_numpy(j.transpose(2,0,1)) for j in flow]
    
    # HD Override if requested
    if hdOverride is not None:
        assert hdOverride.shape == neigbor_hd[0].shape
        neigbor_hd[0] = hdOverride

    # Copy to GPU
    with torch.no_grad():
        input = Variable(input).cuda(gpus_list[0])
        neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
        neigbor_hd = [Variable(j).cuda(gpus_list[0]) for j in neigbor_hd]
        flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]

    # Reshape for model
    input = input.unsqueeze(0)
    bicubic = bicubic.unsqueeze(0)
    target = target.unsqueeze(0)
    neigbor[0] = neigbor[0].unsqueeze(0)
    neigbor_hd[0] = neigbor_hd[0].unsqueeze(0)
    flow[0] = flow[0].unsqueeze(0)

    # Run Model
    t0 = time.time()
    with torch.no_grad():
        prediction = model(input, neigbor, neigbor_hd, flow)
        
    if opt.residual:
        prediction = prediction + bicubic
            
    t1 = time.time()
    
    # Save Images
    save_img(prediction.cpu().data, imgFile, "output")
    save_img(target.cpu().data, imgFile, "target")
    save_img(input.cpu().data, imgFile, "input")

    # Calculate PNSR
    prediction=prediction.cpu()
    ret = prediction.squeeze().unsqueeze(0)
    prediction = prediction.squeeze().numpy().astype(np.float32)
    prediction = prediction*255.
        
    target = target.squeeze().numpy().astype(np.float32)
    target = target*255.

    bicubic = bicubic.squeeze().numpy().astype(np.float32)
    bicubic = bicubic*255.
                
    psnr = PSNR(prediction, target, shave_border=opt.upscale_factor)
    base = PSNR(bicubic, target, shave_border=opt.upscale_factor)

    print("===> Processing: %s || Timer: %.4f sec. || PSNR: %.4f || Base: %.4f" % (imgFile, (t1 - t0), psnr, base))
    return ret, psnr, base

def save_img(img, img_path, subfolder):

    # Convert to Numpy Array
    save_img = np.float32(img.squeeze().clamp(0, 1).numpy())

    # Find save folder and file name
    folder = os.path.basename(os.path.dirname(img_path))
    name = os.path.splitext(os.path.basename(img_path))[0]
    save_dir = os.path.join(os.path.join(opt.output, folder), subfolder)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_fn = os.path.join(save_dir, name + ".png")

    # Use OpenCV for final conversion
    final_img = cv2.cvtColor(save_img, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(save_fn,  save_img*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def PSNR(pred, gt, shave_border=0):
    assert pred.shape == gt.shape
    assert len(pred.shape) == 2
    height, width = pred.shape
    pred = pred[1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    gt = gt[1+shave_border:height - shave_border, 1+shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def processVideo(folder, start):
    psnrs = []
    bases = []

    # Initial Variables
    current = start
    imgFile = os.path.join(folder, "frame%s.jpg" % str(current))
    if not os.path.exists(imgFile):
        return []

    # Save first frame, assume PSNR is perfect
    _, _, _ = processFrame(imgFile, None)

    # Loop through subsequent frames
    current += 1
    pred = None
    imgFile = os.path.join(folder, "frame%s.jpg" % str(current))
    while os.path.exists(imgFile):
        if opt.reset_hd > 0:
            if current % opt.reset_hd == 0:
                print("Reseting HD Frame")
                pred = None
        pred, psnr, base = processFrame(imgFile, pred)
        psnrs.append(psnr)
        bases.append(base)
        if psnr < -250:
            print("PSNR too bad, breaking")
            break
        current += 1
        imgFile = os.path.join(folder, "frame%s.jpg" % str(current))

    return psnrs, bases

def main():
    print("----------------")
    print()
    print("Welcome to the Upscaler!")
    while True:
        print()

        # Get video folder
        num = input("Enter Video # To Process: ")
        folder = os.path.join(opt.data_dir, num)
        if not os.path.exists(folder):
            print("Error: %s doesn't exist" % folder)
            continue

        # Get starting frame
        start = int(input("Enter starting frame #: "))

        psnrs, bases = processVideo(folder, start)

        # Plot PSNRs
        print("Plotting PSNR")
        t = np.arange(len(psnrs)) + 1
        plt.plot(t, psnrs, 'r', label="Model")
        plt.plot(t, bases, 'k', label="Bicubic")
        plt.title("SNR Decay, Video %d" % int(num))
        plt.xlabel("Frame After HD")
        plt.ylabel("Peak SNR (dB)")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()
