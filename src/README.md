# Readme for source code

### Backup CNN
Concerning files: 
1. ```data_reader.py```
2. ```backup_cnn.py```
<br />
Run 1) by typing ```python3 data_reader.py``` .
This file preprocesses the augtmented training data and the test data into .npy files. The labels are one hot arrays with lenght number-of-classes. 
Run 2) by typing ```python3 backup_cnn.py```. 
To train the network run ```python3 backup_cnn.py```. This trains a simple CNN to recognize the characters. See init of the CNN_network class in ```backup_cnn.py``` for the hyper parameters of the network. 
