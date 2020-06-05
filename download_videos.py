# import os

# os.system("cd ~/ssds/Downloads/videos")
# video_id = 0
# os.system("youtube-dl -a video_list.txt -i")


import numpy as np 
import glob, os
os.chdir("/home/zoey/ssds/Downloads/cse599")
import cv2

vid = 0
for video_path in glob.glob("*.mp4"):
	# downsample a video
	print("now processing video: ", video_path)
	save_gray_path = "/home/zoey/ssds/Downloads/gray_scale/{0}".format(vid)
	save_HD_path = "/home/zoey/ssds/Downloads/HD/{0}".format(vid)
	if not os.path.exists(save_gray_path):
		os.makedirs(save_gray_path)
	if not os.path.exists(save_HD_path):
		os.makedirs(save_HD_path)
	vidcap = cv2.VideoCapture(video_path)
	success,image = vidcap.read()
	count = 0

	while success:
		if count>60:
			break
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray_resized = cv2.resize(gray, (int(image.shape[1]/4), int(image.shape[0]/4)), interpolation = cv2.INTER_AREA)
		cv2.imwrite(save_gray_path+"/frame%d.jpg" % count, gray_resized)     # save frame as JPEG file   
		cv2.imwrite(save_HD_path+"/frame%d.jpg" % count, image)     # save frame as JPEG file   

		success,image = vidcap.read()
		count += 1
	vid+=1
