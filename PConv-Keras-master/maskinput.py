import random
import cv2 as cv
import numpy as np

from skimage.feature import canny
from skimage.color import rgb2gray
from matplotlib.pyplot import imread

import torch
import torch.utils.data as data
from torchvision.transforms import ToTensor

class CelebADataset(data.Dataset):
    def __init__(self, width, height, mask, type, train_flist, test_flist, train):
        self.width, self.height = width, height
        self.mask = mask
        self.type = type
        self.train = train
        self.data = self._load_flist(train_flist if train else test_flist)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        with torch.no_grad():
            item = self._load_item(index)
            return item

    def _load_flist(self, flist):
        try:
            return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
        except:
            return Exception('image_path_list error!!!')

    def _load_item(self, index):
        # load item
        image = self._load_image(self.data[index])
        if self.type == 'edge':
            edge = self._load_edge(rgb2gray(image))
        elif self.type == 'mscn':
            edge = self._load_mscn(rgb2gray(image))
        mask = self._load_mask(image)
        return self._to_tensor(image), self._to_tensor(edge), self._to_tensor(mask)

    def _to_tensor(self, x):
        x = ToTensor()(x).float()
        return x

    def _load_image(self, data):
        return np.resize(imread(data), (self.height, self.width, 3))

    def _load_edge(self, gray):
        return canny(gray, sigma=random.randint(1, 4)).astype(np.float)

    def _load_mscn(self, gray):
        kernel1d = cv.getGaussianKernel(5, 3)
        kernel2d = np.outer(kernel1d, kernel1d.transpose())
        mu = cv.filter2D(gray, -1, kernel2d)
        sigma = np.sqrt(np.abs(cv.filter2D(gray * gray, -1, kernel2d) - np.power(mu, 2)))
        mscn = (gray - mu) / (sigma + 1)
        mscn = (mscn - np.min(mscn)) / (np.max(mscn) - np.min(mscn))
        return mscn

    def _load_mask(self, image):
        # c, h, w = image.size()
        h, w, c = image.shape
        m_height, m_width = h // 2, w // 2
        mask = np.zeros((h, w))
        if self.mask == 'block':
            mask_y = np.random.randint(0, h - m_height)
            mask_x = np.random.randint(0, w - m_width)
            mask[mask_y:mask_y + m_height, mask_x:mask_x + m_width] = 1

        elif self.mask == 'blockc':
            mask_y = (h - m_height) // 2
            mask_x = (w - m_width) // 2
            mask[mask_y:mask_y + m_height, mask_x:mask_x + m_width] = 1

        elif self.mask == 'irregular':
            max_width = 20
            if h < 64 or w < 64:
                raise Exception("width and height of mask be at least 64!")
            np.random.seed(1)  ## maskseed = 22, 100 for test
            number = np.random.randint(16, 30)
            for _ in range(number):
                model = np.random.random()
                if model < 0.6: # Draw random lines
                    x1, x2 = np.random.randint(1, h), np.random.randint(1, h)
                    y1, y2 = np.random.randint(1, w), np.random.randint(1, w)
                    thickness = np.random.randint(4, max_width)
                    cv.line(mask, (x1, y1), (x2, y2), (1, 1, 1), thickness)

                elif model >= 0.6 and model < 0.8: # Draw random circle
                    x1, y1 = np.random.randint(1, h), np.random.randint(1, w)
                    radius = np.random.randint(4, max_width)
                    cv.circle(mask, (x1, y1), radius, (1, 1, 1), -1)

                else: # Draw random ellipses
                    x1, y1 = np.random.randint(1, h), np.random.randint(1, w)
                    s1, s2 = np.random.randint(1, h), np.random.randint(1, w)
                    a1 = np.random.randint(3, 180)
                    a2 = np.random.randint(3, 180)
                    a3 = np.random.randint(3, 180)
                    thickness = np.random.randint(4, max_width)
                    cv.ellipse(mask, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)
        else:
            raise
        return mask