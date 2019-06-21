# Sayaret's YOLO and Sliding Window
Repository for the Handwriting Recognition course, 2019

We are the Sayaret team, and this repository contains the code of our dead sea scroll reading system.

## Compile instructions

[TODO]
### Sliding Window

### YOLO

## Task Division
Emile:
* Sliding window for backup_cnn;
* Overall pipeline for yolo (splitting data, augmenting data, etc)
* Data reader for backup_cnn
* Training networks

Roberts:
* Write_to_file

Tomas:
* Preprocessing (all of it);
* Postprocessing

Werner:
* Changing Yolo to train on our data (a lot);
* Post processing (Naive Bayes);
* Overall pipeline for yolo (splitting data, augmenting data, making training lines, training and testing pipeline, etc)
* Finding correct configuration hyperparameters and network layout (largest contribution)
* Training networks

Ruben:
* Initial setup Yolo;
* Backup_cnn & data reader;
* Sliding window approach for backup cnn
* Finding correct configuration hyperparameters and network layout (smaller contribution)
* Overall pipeline for yolo (splitting data, augmenting data, making training lines, training and testing pipeline, etc) (smaller contribution)
* Training networks

## Timeline of Project

| DATE  | TASK                            | PERSON         |
|-------|---------------------------------|----------------|
| 04-17 | Formed group                    | All            |
| 04-18 | Discussion meeting              | All            |
| 04-18 | Create repository               | Werner         |
| 04-23 | Discussion meeting              | All            |
| 04-24 | Start working on binarizer      | Ruben & Tomas  |
| 04-25 | Add data splitter               | Emile          |
| 04-25 | Add morphological operations    | Tomas          |
| 04-25 | Add data augmenter              | Emile & Werner |
| 04-29 | Modify YOLO for custom data     | Werner         |
| 04-30 | Build YOLO train/test pipeline  | Emile & Werner |
| 04-30 | Add image cropping              | Tomas          |
| 05-02 | Finish binarizer                | Tomas          |
| 05-02 | Work on input image shape       | Emile & Werner |
| 05-07 | Add fake line generator         | Werner         |
| 05-08 | Refactor pipeline               | Werner         |
| 05-09 | Improve cropping                | Tomas          |
| 05-09 | Refactor preprocessing          | Tomas          |
| 05-09 | Work on line segmentation       | Ruben          |
| 05-09 | Work on smearing segmentation   | Tomas          |
| 05-09 | Refactor pipeline               | Werner         |
| 05-13 | Refactor preprocessing          | Tomas          |
| 05-14 | Clean up workspace              | Emile & Werner |
| 05-14 | Bugfixing                       | Emile & Ruben  |
| 05-16 | Add preprocessing functionality | Tomas          |
| 05-16 | Prepare code for Google Colab   | Emile          |
| 05-21 | Improve binarization quality    | Tomas          |
| 05-22 | Add augmentation techniques     | Ruben          |
| 05-22 | Clean up preprocessing          | Tomas          |
| 05-22 | Parameter sweeping              | Ruben          |
| 05-27 | Work on acid drop preprocessing | Tomas          |
| 05-27 | Work on improving Yolo          | Werner         |
| 05-28 | Improve acid drop               | Tomas          |
| 05-28 | Improve line generation         | Werner         |
| 05-28 | Improve testing module          | Werner         |
| 05-28 | Bugfixing                       | Ruben          |
| 05-28 | Python to cython                | Tomas          |
| 05-28 | Remove old Yolo weights req.    | Ruben          |
| 06-01 | Create backup CNN               | Ruben          |
| 06-03 | Start working on Yolo filters   | Werner         |
| 06-04 | Convert Yolo to single-channel  | Emile & Werner |
| 06-04 | Improve Yolo normalization      | Ruben          |
| 06-04 | Add Bayesian postprocessing     | Werner         |
| 06-04 | Update & improve preprocessing  | Tomas          |
| 06-05 | Add fake line random padding    | Werner         |
| 06-05 | Bugfixing and code cleaning     | Tomas          |
| 06-05 | Replace Otsu by Sauvola         | Tomas          |
| 06-06 | Yolo parameter sweeping         | Ruben          |
| 06-06 | Improve faulty cropping removal | Tomas          |
| 06-06 | Start on ReadMe                 | Ruben          |
| 06-06 | Start on backup CNN pipeline    | Tomas          |
| 06-06 | Add sliding window approach     | Emile          |
| 06-06 | Add hebrew character writer     | Werner         |
| 06-11 | Improve sliding window peaks    | Ruben          |
| 06-13 | Improve sliding window output   | Ruben          |
| 06-13 | Add many-image functionality    | Tomas          |
| 06-17 | Improve the Yolo filters        | Werner         |
| 06-19 | Fix Yolo normalization bug      | Werner         |
| 06-19 | Improve OS compatibility        | Tomas          |
| 06-19 | Improve backup CNN padding      | Emile          |
| 06-19 | Bugfixes                        | Ruben          |
| 06-19 | Add preprocessing to Yolo       | Tomas          |
| 06-19 | Extend Bayesian postprocessing  | Tomas          |
| 06-19 | Add data augmentation to backup | Ruben          |
| 06-19 | Backup CNN improvements         | Emile & Ruben  |
| 06-20 | Parameter Yolo testing          | Werner         |
| 06-20 | Add file writer functionality   | Roberts        |
| 06-20 | Multiple pipeline improvements  | Ruben          |
| 06-20 | Improve sliding window location | Emile          |
| 06-20 | Clean Yolo code and pipeline    | Werner         |
| 06-21 | Improve final pipeline          | Tomas          |
| 06-21 | Add character acid drop segmen. | Tomas          |
| 06-21 | Yolo parameter sweep            | Werner         |



Article list (for the report references):
https://github.com/YunYang1994/tensorflow-yolov3
