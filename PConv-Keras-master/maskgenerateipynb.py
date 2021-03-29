
import os
import glob
import numpy as np
import cv2
from random import randint, seed
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image



class MaskGenerator():

    def __init__(self, height=512, width=680, channels=3, rand_seed=None, filepath=None):
        """Convenience functions for generating masks to be used for inpainting training

        Arguments:
            height {int} -- Mask height
            width {width} -- Mask width

        Keyword Arguments:
            channels {int} -- Channels to output (default: {3})
            rand_seed {[type]} -- Random seed (default: {None})
            filepath {[type]} -- Load masks from filepath. If None, generate masks with OpenCV (default: {None})
        """

        self.height = 512
        self.width = 680
        self.channels = channels
        self.filepath = filepath

        # If filepath supplied, load the list of masks within the directory
        self.mask_files = []
        if self.filepath:
            filenames = [f for f in os.listdir(self.filepath)]
            self.mask_files = [f for f in filenames if
                               any(filetype in f.lower() for filetype in ['.jpeg', '.png', '.jpg'])]
            print(">> Found {} masks in {}".format(len(self.mask_files), self.filepath))

            # Seed for reproducibility
        if rand_seed:
            seed(rand_seed)

    # 마스크 생성코드
    def _gen_img(self, i):
        img = np.zeros((self.height, self.width, self.channels), np.uint8)

        # Set size scale
        size = int((self.width + self.height) * 0.03)
        if self.width < 64 or self.height < 64:
            raise Exception("Width and Height of mask must be at least 64!")

        # Draw random lines
        for _ in range(randint(1, 20)):
            x1, x2 = randint(1, self.width), randint(1, self.width)
            y1, y2 = randint(1, self.height), randint(1, self.height)
            thickness = randint(3, size)
            cv2.line(img, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        # Draw random circles
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            radius = randint(3, size)
            cv2.circle(img, (x1, y1), radius, (1, 1, 1), -1)

        # Draw random ellipses
        for _ in range(randint(1, 20)):
            x1, y1 = randint(1, self.width), randint(1, self.height)
            s1, s2 = randint(1, self.width), randint(1, self.height)
            a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
            thickness = randint(3, size)
            cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)

        # 흑백 뒤바꾸기
        _mask = 1 - img
        img = img * 255
        plt.imshow(img)
        plt.show()
        # plt.close()
        # plt.savefig('./data/loadroom/mask/mask{}.png'.format(i))
        im = Image.fromarray(img.astype(np.uint8))
        # im.show()
        im.save('./data/loadroom/mask/mask{}.png'.format(i))
        im.close()
        return img, _mask

    def _generate_mask(self):
        """Generates a random irregular mask with lines, circles and elipses"""

        lst_imgs = [i for i in glob.glob("./data/roomtest/*")]

        print(lst_imgs)
        for idx, i in enumerate(lst_imgs):
            img, _mask = self._gen_img(idx)

            image = Image.open(i)
            image = image.resize((680, 512), Image.ANTIALIAS)
            # plt.imshow(image)
            # plt.show()
            i_mage = np.array(image, dtype='uint')
            i_magee = i_mage * _mask + img
            plt.imshow(i_magee)
            plt.show()
            # plt.savefig('./data/loadroom/output/output{}.png'.format(idx))
            im = Image.fromarray(i_magee.astype(np.uint8))
            # im.show()
            im.save('./data/loadroom/output/output{}.png'.format(idx))
            im.close()

        # return img


mask = MaskGenerator()
mask._generate_mask()

