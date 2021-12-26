import os
import glob
from PIL import Image
import cv2
import re
from sklearn.model_selection import train_test_split

path = './data/training_hr_images/'

traindir = './data/train/'
valdir = './data/val/'

if not os.path.isdir(traindir):
    os.mkdir(traindir)
if not os.path.isdir(valdir):
    os.mkdir(valdir)

'''
images = []

for dir_path, dir_names, file_names in os.walk(path):
    for f in file_names:
        images.append(os.path.join(dir_path, f))
'''

print(path)
imageallnames = sorted(glob.glob(path+"*.png"),
    key=lambda x:[int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)]
)

imagename=[]
for name in imageallnames:
    filename = name.split("/")[3]
    imagename.append(filename)

train_img, val_img = train_test_split(imagename, test_size=0.2)

print(traindir+train_img[0])


for trainpath in train_img:
    img = cv2.imread(path+trainpath)
    cv2.imwrite(traindir+trainpath,img)
for valpath in val_img:
    img = cv2.imread(path+valpath)
    cv2.imwrite(valdir+valpath,img)