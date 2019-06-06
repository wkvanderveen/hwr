# Preprocessing pipeline running instructions

### Install the Cython library using pip

```
pip install cython
```

Cython compiles to c++ code, so a c++ compiler is needed on the system to compile the code.
Rules for this are given in the ```setup.py``` file. It is preconfigured to work on Windows, but might work as is on Linux and Mac. The cython code itself should work regardless of platform.

### Compile the preprocessing pipeline

Move to the preprocessing directory
```
cd preprocessing/
```

Compile the Cython code
```
python setup.py build_ext --inplace
```
This command also noted at the top of the ```setup.py``` file. 
The code is compiled into ```.pyd``` files (on windows, this might differ on linux)

### Test the preprocessing pipeline

An example call to the pipeline is given in ```src/preprocessing_example.py```.
If Cython was correctly installed and the code compiled the example program will execute without errors.
```
cd ..
python preprocessing_example.py
```
The example program takes every file in ```hwr/data/image-data``` and converts it into binarized lines. The lines are then saved in ```hwr/data/lines```.
These paths can be changed at the top of the ```src/preprocessing_example.py``` file.


