# Before running, ensure that the data folder only contains the "letters" data (with subfolders for each letter)

# Create augmented dataset from original data, and anchors and labels
python3 augment_data.py

# Convert augmented data into large .tfrecord files. Creates train and test folders.
cat ../../data/dataset.txt | head -n  55000 > ../../data/dataset_train.txt
cat ../../data/dataset.txt | tail -n +55001 > ../../data/dataset_test.txt
python3 core/convert_tfrecord.py --dataset_txt ../../data/dataset_train.txt --tfrecord_path_prefix ../../data/dataset_train
python3 core/convert_tfrecord.py --dataset_txt ../../data/dataset_test.txt  --tfrecord_path_prefix ../../data/dataset_test

# (Optional) Show an input image (in the code you can define which one)
# python3 show_input_image.py

# Get prior anchors and rescale the values to the range [0,1]
python3 kmeans.py

# Obtain the pretrained ImageNet weights for the network
python3 convert_weight.py --convert

# Train the network. Checkpoints at every 100th step. Validation at every 50th step.
# Keep an eye on overfitting (e.g. through TensorBoard)
python3 quick_train.py

#####################################

