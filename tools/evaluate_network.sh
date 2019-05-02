python3 convert_weight.py -cf ../../data/checkpoint/yolov3.ckpt-600 -nc 27 -ap ../../data/anchors.txt --freeze

# Test the network
python3 quick_test.py

# Evaluate the performance
python3 evaluate.py
