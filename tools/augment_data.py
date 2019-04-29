import os
import pathlib
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

data_location = '../../data/letters/'  # original data
data_destination = '../../data/augmented/letters/'  # augmented data

# Make goal directory if it doesn't already exist
pathlib.Path(data_destination).mkdir(parents=True, exist_ok=True)

# Create text labels
f = open("../../data/dataset.txt", "w+")
f = open("../../data/anchors.txt", "w+")

img_h, img_w = 32, 32

# MAKE BLUR OPS
blur_degrees = [2, 4, 6]
blur_ops = []
for blur_degree in blur_degrees:
    blur_ops.append(iaa.Sequential([
        iaa.GaussianBlur(sigma=(0, blur_degree)) # blur images with a sigma of 0 to 3.0
    ]))

# MAKE NOISE OPS
noise_degrees = [0.1, 0.3, 0.5]
noise_ops = []
for noise_degree in noise_degrees:
    noise_ops.append(iaa.Sequential([
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, noise_degree*255), per_channel=0.5) # blur images with a sigma of 0 to 3.0
    ]))

# MAKE SALT OPS
salt_degrees = [0.1, 0.3, 0.5]
salt_ops = []
for salt_degree in salt_degrees:
    salt_ops.append(
        iaa.arithmetic.CoarseSalt(p=salt_degrees, size_percent=(10,10)))

def write_to_dataset(path,x1,x2,y1,y2,c):
    with open("../../data/dataset.txt", "a") as filename:
        print(f"{path} {x1-1} {x2-1} {y1} {y2} {c-1}", file=filename)

for letter_dir, dirs, imgs in os.walk(data_location):

    # Skip if in root directory. Also count number of letters
    if dirs:
        idx=0
        num_letters=len(dirs)
        continue
    idx += 1

    # Make augmented letter directory if it doesn't exist
    _, letter = os.path.split(letter_dir)
    new_dir = os.path.join(data_destination, letter)
    pathlib.Path(new_dir).mkdir(parents=True, exist_ok=True)

    print("Processing letter {} ({}/{})...".format(letter, idx, num_letters))

    for img_idx, img in enumerate(imgs):
        img_path = os.path.join(letter_dir, img)
        original = cv2.imread(img_path)
        original = cv2.resize(original, (img_w,img_h))

        # WRITE ORIGINAL
        aug_path = os.path.join(new_dir, letter+str(img_idx)+"-ORIG"+".jpeg")
        cv2.imwrite(aug_path, original)
        write_to_dataset(aug_path, 0, 0, img_w,img_h,idx)

        # DO BLURRING
        for op_idx, operation in enumerate(blur_ops):
            aug_path = os.path.join(new_dir, letter+str(img_idx)+"-BLUR"+str(op_idx)+".jpeg")
            aug = operation.augment_image(original)
            cv2.imwrite(aug_path, aug)
            write_to_dataset(aug_path, 0, 0, img_w,img_h,idx)

        # DO NOISE
        for op_idx, operation in enumerate(noise_ops):
            aug_path = os.path.join(new_dir, letter+str(img_idx)+"-NOISE"+str(op_idx)+".jpeg")
            aug = operation.augment_image(original)
            cv2.imwrite(aug_path, aug)
            write_to_dataset(aug_path, 0, 0, img_w,img_h,idx)

        # DO SALT
        for op_idx, operation in enumerate(salt_ops):
            aug_path = os.path.join(new_dir, letter+str(img_idx)+"-SALT"+str(op_idx)+".jpeg")
            aug = operation.augment_image(original)
            cv2.imwrite(aug_path, aug)
            write_to_dataset(aug_path, 0, 0, img_w,img_h,idx)
