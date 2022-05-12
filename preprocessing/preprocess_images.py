import torch, os, pandas
from torchvision import transforms
from PIL import Image

def crop_image(image, size):
    return transforms.functional.crop(image, 0, 0, size[0], size[1])

def resize_image(image):
    return image.resize((int(image.size[0]/2), int(image.size[1]/2)))

def pad_image(IMAGE):
    right = 8
    left = 8
    top = 8
    bottom = 8
    width, height = IMAGE.size
    new_width = width + right + left
    new_height = height + top + bottom
    return (Image.new(IMAGE.mode, (new_width, new_height), (255,255, 255)))

def preprocess_images(img_batch, datapath):
    """
    RuntimeError: only Tensors of floating point dtype can require gradients    Crop, padding, and downsample the image.
    :params img_batch: batch of images
    :return: processed images
    """

    images_dict = {}
#    new_img_batch = []
    for image_label in img_batch:
        # opening the image
        # image_label = str(image_label.cpu().numpy())
        IMAGE = Image.open(os.path.join(datapath, f'{image_label}.png'))

        # crop the image
        IMAGE = crop_image(IMAGE, [600,60])
        # resize
        IMAGE = resize_image(IMAGE)
        # padding
        IMAGE = pad_image(IMAGE)
        # convert to tensor
        convert = transforms.ToTensor()
        IMAGE = convert(IMAGE)

#        new_img_batch.append(IMAGE)
        images_dict[f'image_label'] = IMAGE

        df = pandas.DataFrame(images_dict, columns=['ID','IMAGES'])
        df.to_csv('data/images_tensor.csv', index=True)

#    return (new_img_batch)
