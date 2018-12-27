import torch.utils.data as data
import pandas as pd
import numpy as np

import os
import os.path
import sys
import ast
import cv2

BASE_SIZE = 256
LW = 6
size = 64


def make_dataset(dir, ks):
    images = []
    n = 1
    L = ks[1]-ks[0]
    for k in range(ks[0], ks[1]):
        d = os.path.join(dir, 'train_k{}.csv'.format(k))
        f = pd.read_csv(d)
        
        for i in range(len(f)//300):
            target = f.iloc[i].y
            item = (d, i, target)
            images.append(item)
            
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %d%%" % ('='*((50*n)//L), 100*n/L))
        sys.stdout.flush()
        n += 1

    return images


def draw_cv2(raw_strokes, size = BASE_SIZE, lw = LW, time_color = True):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            color = 255 - min(t, 10) * 13 if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img
    
def squarify(img):
    img = img[~np.all(img == 0, axis=1)]
    img = img[:,~np.all(img == 0, axis=0)]
    (a, b)= img.shape
    if a > b:
        c = (a - b) // 2
        padding=((0,0),(c ,a-b- c))
    else:
        c = (b-a) // 2
        padding=((c, b-a - c), (0,0))
    return np.pad(img,padding,mode='constant',constant_values=0)
    
def draw_cv2_sq(raw_strokes, size = BASE_SIZE, lw = LW, time_color = True, num = 1e3):
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for t, stroke in enumerate(raw_strokes):
        if t == num:
            break
        for i in range(len(stroke[0]) - 1):
#             color = 255 - min(t, 10) * 13 if time_color else 255
            color = 255 - min(t, 10) * 12 - int(i/len(stroke[0])*12.0) if time_color else 255
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                         (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
    img = squarify(img)
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size), interpolation = cv2.INTER_NEAREST)
    else:
        return img

def image_generator(path, row, size = size, lw = LW, time_color = True):
    f = pd.read_csv(path)
    raw_strokes = eval(f.iloc[row].drawing)
    img = np.zeros((size, size, 1))
    img[:,:,0] = draw_cv2_sq(raw_strokes, size=size, lw=lw, time_color=time_color)
    return img.astype(np.float32)

def image_loader(strokes, size = size, lw = LW, time_color = True):
    raw_strokes = eval(strokes)
    img = np.zeros((size, size, 3))
    img[:,:,0] = draw_cv2_sq(raw_strokes, size=size, lw=lw, time_color=time_color)
    img[:,:,1] = img[:,:,0]
    img[:,:,2] = img[:,:,0]
    return img.astype(np.float32)

def image_loader3(strokes, size = size, lw = LW, time_color = True):
    raw_strokes = eval(strokes)
    img = np.zeros((size, size, 3))
    img[:,:,0] = draw_cv2_sq(raw_strokes, size=size, lw=lw, time_color=time_color)
    # img[:,:,1] = draw_cv2_sq(raw_strokes, size=size, lw=lw, time_color=time_color, num=5)
    img[:,:,1] = draw_cv2_sq(raw_strokes, size=size, lw=lw, time_color=time_color)
    img[:,:,2] = draw_cv2_sq(raw_strokes, size=size, lw=lw, time_color=time_color, num=1)
    return img.astype(np.float32)


class Dataset_csv_ks(data.Dataset):

    def __init__(self, root, ks, size = size, loader = image_generator, transform = None, target_transform = None):
        samples = make_dataset(root,ks)
        self.loader = loader
        self.samples = samples
        self.size = size
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, row, target = self.samples[index]
        sample = self.loader(path, row, size=self.size)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
    def __len__(self):
        return len(self.samples)
    
class Dataset_csv_new(data.Dataset):

    def __init__(self, root, size = size, lw = LW, loader = image_loader3,
                 key = False, shuffle = True, transform = None, target_transform = None):
        samples = pd.read_csv(root)
        if shuffle:
            samples = samples.sample(frac=1).reset_index(drop=True)
        self.loader = loader
        self.samples = samples
        self.size = size
        self.lw = lw
        self.key = key
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        strokes, target = self.samples.iloc[index].drawing, self.samples.iloc[index].y
        sample = self.loader(strokes, size=self.size, lw=self.lw)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.key:
            return sample, target, self.samples.iloc[index].key_id
        return sample, target
    
    def __len__(self):
        return len(self.samples)
    
class Dataset_csv_test(data.Dataset):

    def __init__(self, root, size = size, lw = LW, loader = image_loader3, transform = None):
        samples = pd.read_csv(root)
        self.loader = loader
        self.samples = samples
        self.size = size
        self.lw = lw
        self.transform = transform

    def __getitem__(self, index):
        strokes, key_id = self.samples.iloc[index].drawing, self.samples.iloc[index].key_id
        sample = self.loader(strokes, size=self.size, lw=self.lw)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, key_id
    
    def __len__(self):
        return len(self.samples)
