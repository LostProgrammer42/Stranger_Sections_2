import os
import numpy as np
from PIL import Image, ImageFilter

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

image_folder = '/media/umic/my_label/stranger_sections_2/stranger-sections-2-starter-notebook/training_images/image'
mask_folder = '/media/umic/my_label/stranger_sections_2/stranger-sections-2-starter-notebook/label'
images = sorted(os.listdir(image_folder))
masks = sorted(os.listdir(mask_folder))
#Loop through all images in given folder
for i in range(len(images)):
        #Load images and masks
        img_name = os.path.join(image_folder, images[i])
        mask_name = os.path.join(mask_folder, masks[i]) 
        image = Image.open(img_name)
        mask = np.load(mask_name)
         
        #Pad images and masks to make them 1360x1360
        image = add_margin(image,168,0,168,0,(0,0,0))
        mask = np.pad(mask, ((168,168),(0,0)), 'constant', constant_values=0)

        #Preprocessing
        #1.) Make images grayscale, no changes to mask while grayscaling
        # image = image.convert('L')

        #2.) Apply median filter
        image = image.filter(ImageFilter.MedianFilter(size = 5))  

        #Augment the images, masks
        #1.) Rotation by 90 , 180, 270 degrees and saving
        image.save(f'/media/umic/my_label/stranger_sections_2/augmented_images/{images[i]}') #Saving original one
        np.save(f'/media/umic/my_label/stranger_sections_2/augmented_masks/{masks[i]}',mask) # Saving original one

        image = image.rotate(90)
        mask = np.rot90(mask)
        image.save(f'/media/umic/my_label/stranger_sections_2/augmented_images/90_Rotated_{images[i]}') #Saving 90 degrees rotated image
        np.save(f'/media/umic/my_label/stranger_sections_2/augmented_masks/90_Rotated_{masks[i]}',mask) # Saving 90 degrees rotated mask

        image = image.rotate(90)
        mask = np.rot90(mask)
        image.save(f'/media/umic/my_label/stranger_sections_2/augmented_images/180_Rotated_{images[i]}') #Saving 180 degrees rotated image
        np.save(f'/media/umic/my_label/stranger_sections_2/augmented_masks/180_Rotated_{masks[i]}',mask) # Saving 180 degrees rotated mask

        image = image.rotate(90)
        mask = np.rot90(mask)
        image.save(f'/media/umic/my_label/stranger_sections_2/augmented_images/270_Rotated_{images[i]}') #Saving 270 degrees rotated image
        np.save(f'/media/umic/my_label/stranger_sections_2/augmented_masks/270_Rotated_{masks[i]}',mask) # Saving 270 degrees rotated mask

        #2.) Flipping horizontally and then rotating
        #Load original
        image = Image.open(img_name)
        mask = np.load(mask_name)
        image = add_margin(image,168,0,168,0,(0,0,0))
        image = image.filter(ImageFilter.MedianFilter(size = 5))
        mask = np.pad(mask, ((168,168),(0,0)), 'constant', constant_values=0)

        #Flipping horizontally
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = np.fliplr(mask)

        image.save(f'/media/umic/my_label/stranger_sections_2/augmented_images/Flipped_{images[i]}') #Saving flipped one
        np.save(f'/media/umic/my_label/stranger_sections_2/augmented_masks/Flipped_{masks[i]}',mask) # Saving flipped one

        image = image.rotate(90)
        mask = np.rot90(mask)
        image.save(f'/media/umic/my_label/stranger_sections_2/augmented_images/90_Rotated_Flipped_{images[i]}') #Saving 90 degrees rotated image
        np.save(f'/media/umic/my_label/stranger_sections_2/augmented_masks/90_Rotated_Flipped_{masks[i]}',mask) # Saving 90 degrees rotated mask

        image = image.rotate(90)
        mask = np.rot90(mask)
        image.save(f'/media/umic/my_label/stranger_sections_2/augmented_images/180_Rotated_Flipped_{images[i]}') #Saving 180 degrees rotated image
        np.save(f'/media/umic/my_label/stranger_sections_2/augmented_masks/180_Rotated_Flipped_{masks[i]}',mask) # Saving 180 degrees rotated mask

        image = image.rotate(90)
        mask = np.rot90(mask)
        image.save(f'/media/umic/my_label/stranger_sections_2/augmented_images/270_Rotated_Flipped_{images[i]}') #Saving 270 degrees rotated image
        np.save(f'/media/umic/my_label/stranger_sections_2/augmented_masks/270_Rotated_Flipped_{masks[i]}',mask) # Saving 270 degrees rotated mask









    