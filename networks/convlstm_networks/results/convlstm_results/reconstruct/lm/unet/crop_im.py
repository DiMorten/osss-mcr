import cv2
from icecream import ic
im = cv2.imread('label_t_mar_unet_openmax.png')
ic(im.shape)
#im = im[4000:6000,4000:]
#im = im[5000:6000,4000:]
#im = im[5000:6000,4000:6000]

#im = im[5000:6000,4500:6000]
#im = im[5000:6000,4700:5700]
#im = im[5000:6000,4700:5800]
#im = im[5300:6000,4700:5800]
#im = im[5200:6000,5000:5800]
im = im[5200:6100,4900:5800]

ic(im.shape)
cv2.imwrite('cropped_im.png', im)