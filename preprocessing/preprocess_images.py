import torch, os, pandas, numpy, math
import multiprocessing
import matplotlib.pyplot as plt
from multiprocessing import Pool, Lock, TimeoutError
from torchvision import transforms
from PIL import Image

# finding mean width and height for this batch
# Ws, Hs = [], []

# final image tensor for current batche
# new_img_batch = []

def crop_image(image, size):
    return transforms.functional.crop(image, 0, 0, size[0], size[1])

def resize_image(image, mean_size):
    # return image.resize((int(image.size[0]/2), int(image.size[1]/2)))
    return image.resize((mean_size[0], mean_size[1]))

def pad_image(IMAGE):
    right = 8
    left = 8
    top = 8
    bottom = 8
    width, height = IMAGE.size
    new_width = width + right + left
    new_height = height + top + bottom
    # #return (Image.new(IMAGE.mode, (new_width, new_height), (255,255, 255)))
    result = Image.new(IMAGE.mode, (new_width, new_height))
    result.paste(IMAGE,(left, top))
    return result

def mean_w_h(img_batch):
    '''
    finding the mean width and height of images for this batch
    '''
    # global Ws, Hs

    Ws, Hs = [], []
    for _i in img_batch:
        _I = Image.open(os.path.join('data/images', f'{_i}.png'))
        _w, _h = _I.size
        Ws.append(_w)
        Hs.append(_h)
        mean_w, mean_h = numpy.mean(Ws), numpy.mean(Hs)
    return mean_w, mean_h


def preprocess_images(img_batch):
    """
    RuntimeError: only Tensors of floating point dtype can require gradients
    Crop, padding, and downsample the image.
    We will modify the images according to the max size of the image/batch

    :params img_batch: batch of images
    :return: processed image tensor for enitre batch-[Batch, Channels, W, H]
    """

    # global new_img_batch, Ws, Hs

    # if want to print full tensors
    # torch.set_printoptions(profile="full")

    # mean width and height
    mean_w, mean_h = mean_w_h(img_batch)

    new_img_batch = []
    for idx, image_label in enumerate(img_batch):
        # opening the image
        IMAGE = Image.open(os.path.join('data/images', f'{image_label}.png'))

        # resize
        IMAGE = resize_image(IMAGE, (math.ceil(mean_w), math.ceil(mean_h)))

        # crop the image
        # IMAGE = crop_image(IMAGE, [max_h+10, max_w+50])
        # if idx==2:
        #     plt.imshow(IMAGE)
        #     plt.show()

        # padding
        IMAGE = pad_image(IMAGE)

        # convert to tensor
        convert = transforms.ToTensor()
        IMAGE = convert(IMAGE)

        # appending the final image tensor
        new_img_batch.append(IMAGE)

    return (new_img_batch)

#
# def main():
#     # finding mean width and height
#     with Pool(multiprocessing.cpu_count()-10) as pool:
#         pool.map(mean_w_h, img_batch)
#
#     with Pool(multiprocessing.cpu_count()-10) as pool:
#         pool.map(parallel_preprocess_images, img_batch)
#
#
# # def preprocess_images(img_batch):
# if __name__ == "__main__":
#     main()

    # return new_img_batch
