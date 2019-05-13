# Before running, ensure that the data folder contains the "letters" data (with subfolders for each letter)

# # Split the data into a training and testing letter dataset
# python3 split.py

# # Augment the training letters
# python3 data_augmenter.py

# # Make training and testing lines from the respective letter sets. Also make labels.
# python3 construct_train_images.py

# # Convert the lines into large .tfrecord files.
# python3 core/convert_tfrecord.py --dataset_txt ../../data/labels-train.txt --tfrecord_path_prefix ../../data/lines-train
# python3 core/convert_tfrecord.py --dataset_txt ../../data/labels-test.txt  --tfrecord_path_prefix ../../data/lines-test

# # Get prior anchors and rescale the values to the range [0,1]
# python3 kmeans.py

# OR do all previous steps using main
python3 main.py

# (Optional) Show an input image (in the code you can define which one)
# python3 show_input_image.py

# Obtain the pretrained ImageNet weights for the network
python3 convert_weight.py --convert

# # Train the network. Checkpoints at every 100th step. Validation at every 50th step.
# # Keep an eye on overfitting (e.g. through TensorBoard)
python3 quick_train.py

# #####################################

