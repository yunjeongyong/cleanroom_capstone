import cv2
import numpy as np

from Sketcher import Sketcher
from libs.util import MaskGenerator, ImageChunker
from libs.pconv_model import PConvUnet

import sys
from copy import deepcopy

print('load model...')
model = PConvUnet(vgg_weights=None, inference_only=True)
model.load('pconv_imagenet.h5', train_bn=False)
# model.summary()

# sys.argv.append('data/images/04.jpg')--> run으로 실행가능. 
img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)

img_masked = img.copy()
mask = np.zeros(img.shape[:2], np.uint8)

# 이미지의 sketcher라는 클래스를 opencv에서 가져옴. 마음대로 이미지에 색칠을 할 수 있게 만들음. 
sketcher = Sketcher('image', [img_masked, mask], lambda : ((255, 255, 255), 255))
chunker = ImageChunker(512, 512, 30) # 이미지 chunker라는 클래스로 이미지를 쪼갰다가 다시 합치는 과정

while True:
  key = cv2.waitKey()

  if key == ord('q'): # quit
    break
  if key == ord('r'): # reset
    print('reset')
    img_masked[:] = img
    mask[:] = 0
    sketcher.show()
  if key == 32: # hit spacebar to run inpainting 스페이스바를 누르면 
    input_img = img_masked.copy() # 마스크된 이미지를 넣어주고 
    input_img = input_img.astype(np.float32) / 255. # 학습되었기 때문에 255로 나눠준다. 

    # 마스크는 0-1사이의 숫자로 넣어야 함. 지우고 싶은 부분이 0, 안지우고 그냥 냅두고 싶음부분이 1
    input_mask = cv2.bitwise_not(mask) 
    input_mask = input_mask.astype(np.float32) / 255. # 0~255로 되어있으니까 0~1로 만들어주기 위해서. 
    input_mask = np.repeat(np.expand_dims(input_mask, axis=-1), repeats=3, axis=-1) # 마스크를 만들었을때 2차원 array로 되어있는데 
    # 이것을 3차원 3채널 array로 만들어주기 위해서 

    # cv2.imshow('input_img', input_img)
    # cv2.imshow('input_mask', input_mask)

    print('processing...')

    # 모델의 인풋이미지는 원래 512x512가 들어와야하는데, 이미지가 512보다 큰 경우 이미지를 자른다. 이미지 자르고 512 사이즈로 하나씩 넣어준다. 
    # 모델을 돌리기 전에는 dimension_preprocess를 두개를 해서 모델에다가 넣는다. 
    chunked_imgs = chunker.dimension_preprocess(deepcopy(input_img))
    chunked_masks = chunker.dimension_preprocess(deepcopy(input_mask)) # 자른 이미지를 다시 모델에 하나씩 넣어서 다시 합쳐주는 chunker클래스

    # for i, im in enumerate(chunked_imgs):
    #   cv2.imshow('im %s' % i, im)
    #   cv2.imshow('mk %s' % i, chunked_masks[i])

    pred_imgs = model.predict([chunked_imgs, chunked_masks])
    result_img = chunker.dimension_postprocess(pred_imgs, input_img) # 안풋이미지는 복원하기 위해서 넣어주는 것. 

    print('completed!')

    cv2.imshow('result', result_img)

cv2.destroyAllWindows()
