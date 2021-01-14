# essentially to show how to get to where to crop
# result: crop from (54, 35) to (390, 253)

from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt

# import example spectrogram in grayscale as testim_gray

# convert to np array, get shape (it is (288, 432))
data_gray = asarray(testim_gray)
print(data_gray.shape)

# finding where to crop
# does not work perfectly, some manual trial anf error needed

left_crop = 0
for p in data_gray[58]:
    if p == 255:
        left_crop += 1
    else:
        left_crop += 1
        break

print(left_crop)


right_crop = 0
for p in data_gray[58][::-1]:
    if p == 255:
        right_crop += 1
    else:
        right_crop += 1
        break

print(right_crop)


# rotate
grayim = Image.fromarray(data_gray)
grayimrot = grayim.transpose(Image.ROTATE_90)
# top is now left, bottom is right
datagrayrot = asarray(grayimrot)

top_crop = 0
for p in datagrayrot[58]:
    if p == 255:
        top_crop += 1
    else:
        top_crop += 1
        break

print(top_crop)

bottom_crop = 0
for p in datagrayrot[37][::-1]:
    if p == 255:
        bottom_crop += 1
    else:
        bottom_crop += 1
        break

print(bottom_crop)


# cropping attempt
cropped = grayim.crop((54, 35, 390, 253))
cropped_data = asarray(cropped)
print(cropped_data.shape)

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(cropped)
plt.show()
