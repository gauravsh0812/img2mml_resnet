import torch, os
import matplotlib.pyplot as plt
from multiprocessing import Pool, Lock, TimeoutError
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image


# finding mean width and height for this batch
# Ws, Hs = [], []

# final image tensor for current batche
new_img_batch = []

def crop_image(image, size):
    return transforms.functional.crop(image, 0, 0, size[0], size[1])

def resize_image(image):
    return image.resize((int(image.size[0]/2), int(image.size[1]/2)))
    # return image.resize((mean_size[0], mean_size[1]))

def pad_image(IMAGE):
    right = 8
    left = 8
    top = 8
    bottom = 8
    width, height = IMAGE.size
    new_width = width + right + left
    new_height = height + top + bottom
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
        _I = Image.open(os.path.join(f'data/images/{_i}'))
        _w, _h = _I.size
        Ws.append(_w)
        Hs.append(_h)

    return Ws, Hs

def preprocess_images(images):
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
    # mean_w, mean_h = mean_w_h(img_batch)
    mean_w, mean_h = 500, 50

    # new_img_batch = {}
    for idx, image_label in enumerate(images):

        if idx%10000==0:print(idx)
        # opening the image
        IMAGE = Image.open(os.path.join('data/images', f'{image_label}'))

        # crop the image
        IMAGE = crop_image(IMAGE, [mean_h, mean_w])

        # resize
        IMAGE = resize_image(IMAGE)#, (math.ceil(mean_w), math.ceil(mean_h)))

        # padding
        IMAGE = pad_image(IMAGE)

        # convert to tensor
        convert = transforms.ToTensor()
        IMAGE = convert(IMAGE)
        
        save_image(IMAGE, f'data/image_tensors/{image_label}')

        # appending the final image tensor
        # new_img_batch[image_label.split('.')[0]]=IMAGE


    # return (new_img_batch)

def main():

    images = os.listdir('data/images')
    preprocess_images(images)

    '''
    # finding mean width and height
    with Pool(multiprocessing.cpu_count()-5) as pool:
        pool.map(mean_w_h, images)

    # plotting

    w_dict, h_dict = {}, {}
    for threshold in range(0, 80, 5):
        for (w,h) in zip(Ws, Hs):
            if threshold <=h< threshold+5:
                if f'{threshold}-{threshold+5}' in h_dict.keys():
                    h_dict[f'{threshold}-{threshold+5}']+=1
                else: h_dict[f'{threshold}-{threshold+5}']=1
            #if threshold <=h< threshold+2:
            #    if f'{threshold}-{threshold+}' in h_dict.keys():
            #        h_dict[f'{threshold}-{threshold+20}']+=1
            #    else:
            #        h_dict[f'{threshold}-{threshold+20}']=1

    # print(w_dict)
    plt.bar(list(h_dict.keys()), h_dict.values(), color='b')
    plt.savefig('data/h_dict.png')

    #plt.show()

    '''

    # img_df = pandas.DataFrame(new_img_batch, columns=['ID','IMG'])
    # img_df.to_csv('data/images_tensor.csv', index=True)


if __name__ == "__main__":
    main()
