# Genetics Algorithm Ensemble Stacking for Chest X-Ray dataset
This is the source code for the course "Tối ưu hóa và ứng dụng" - DS106.P11, inspired by this paper https://www.sciencedirect.com/science/article/pii/S1746809423009394?ref=pdf_download&fr=RR-2&rr=8efa145a1cfc5fcd

with the Chest X-Ray dataset from 
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?fbclid=IwY2xjawHEkfJleHRuA2FlbQIxMAABHRkjWEbT6p2WepkKcSFhlyy8uYCo2OlRGUHAQXitQeCXTb1fbfI7X6TLcw_aem_88coY5eBKfjaQ6c5Ev1swQ

This implementation managed to achieve an F1-score of 0.875 on test set.

## Requirements
Create a conda env with python 3.10, then `pip install -r requirements.txt`

## Introduction
This implementation is divided into 3 stages (`stage_1.py, stage_2.py, stage_3.py`), in which the algorithm is detailed inside the docstring of each file.

The purpose of each folder in this workspace is as follows:

**data/**: Contain *chest_x_ray_data.npz* which is the compressed dataset (having x_train, y_train, x_val, y_val, x_test and y_test) which will be loaded by the **load_data** function in `process_data.py`.

**meta/**: Contain *x_train_meta.npy* with shape (number_of_rows_in_train_dataset, number_of_pretrained_models) which represents the 5-fold cross-validation prediction on the train dataset. Further details are documented in `stage_1.py`.

**pretrained/**: Contain 11 pretrained model files (ML models will have **.pkl** format and CNN models will have **.keras** format). Available models are listed in `__init__.py`.

**cache/**: Contain *cache_predictions.npy*, which should be the test set predictions of the above 11 pretrained models. Details in `stage_2.py`.

You can also download and unzip the file **archive.zip** (which already prepared these 4 folders) in this directory from the link below in case rerunning the entire baseline is not needed: https://drive.google.com/file/d/1FifD8AasWbzHgiFreNYfFB1alxz6aP5H/view?usp=sharing

## How to run
Just run the stages `stage_1.py, stage_2.py, stage_3.py` sequentially, that's it.

**Note**: The codes in `geneetic_algorithm.py` and `chromosome.py` could be used as algorithm templates for other projects. You may need to change the implementation of the Chromosome class to fit the logic of your problem.


