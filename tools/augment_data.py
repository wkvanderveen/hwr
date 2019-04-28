import os
import pathlib
import cv2
import imgaug as ia
from imgaug import augmenters as iaa

data_location = '../../data/letters/'  # original data
data_destination = '../../data/augmented/letters/'  # augmented data

# Make goal directory if it doesn't already exist
pathlib.Path(data_destination).mkdir(parents=True, exist_ok=True)

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
    salt_ops.append(iaa.arithmetic.CoarseSalt(p=salt_degrees, size_percent=(10,10)))

for letter_dir, dirs, imgs in os.walk(data_location):

    # Skip if in root directory
    if dirs: continue

    # Make augmented letter directory if it doesn't exist
    _, letter = os.path.split(letter_dir)
    new_dir = os.path.join(data_destination, letter)
    pathlib.Path(new_dir).mkdir(parents=True, exist_ok=True)

    for img_idx, img in enumerate(imgs):
        img_path = os.path.join(letter_dir, img)
        print("Augmenting {}".format(img_path))
        original = cv2.imread(img_path)

        # WRITE ORIGINAL
        aug_path = os.path.join(new_dir, letter+str(img_idx)+"-ORIG"+".jpeg")
        cv2.imwrite(aug_path, original)

        # DO BLURRING
        for op_idx, operation in enumerate(blur_ops):
            aug_path = os.path.join(new_dir, letter+str(img_idx)+"-BLUR"+str(op_idx)+".jpeg")
            aug = operation.augment_image(original)
            cv2.imwrite(aug_path, aug)

        # DO NOISE
        for op_idx, operation in enumerate(noise_ops):
            aug_path = os.path.join(new_dir, letter+str(img_idx)+"-NOISE"+str(op_idx)+".jpeg")
            aug = operation.augment_image(original)
            cv2.imwrite(aug_path, aug)

        # DO SALT
        for op_idx, operation in enumerate(salt_ops):
            aug_path = os.path.join(new_dir, letter+str(img_idx)+"-SALT"+str(op_idx)+".jpeg")
            aug = operation.augment_image(original)
            cv2.imwrite(aug_path, aug)




"""







class DataReader:
    def __init__(self):
        self.data = []
        self.path = '../../data/letters/' #define path of example letter images
        self.save_path = '../../data/'
        self.save_file = self.save_path + "letters"
        self.threshold = 200

    def read_letters(self):
        letters = []
        print("Reading letters from dataset")
        for filepath in os.listdir(self.path):
            sub_path = self.path+filepath
            for sub_file in os.listdir(sub_path):
                img_path = sub_path+'/'+sub_file
                #print(img_path)
                letters.append(plt.imread(img_path))
        letters = self.binarize_images(np.array(letters))
        print("Number of letters found in dataset:", letters.shape)
        np.save(self.save_file, letters)
        print("Letters saved as npy file: ", self.save_file)

    def binarize_images(self, data):
        binarized_images = []
        for image in data:
            binary_image = np.where(image>self.threshold, 1, 0)
            binarized_images.append(binary_image)
        return np.array(binarized_images)

    def read_test_data(self):
        # Read the test data and call the preprocessing function here
        pass


reader = DataReader()
reader.read_letters()
#reader.binarize_images()
"""
