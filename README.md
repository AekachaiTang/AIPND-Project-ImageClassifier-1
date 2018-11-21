# [AIPND] Project: Image Classifier

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Environment

* Numpy
* Pandas
* Matplotlib
* jupyter notebook
* torchvision
* PyTorch

## Jupyter Notebook
* Image Classifier Project.ipynb

## Command Line Application
* Train a new network on a data set with ```train.py```
  * Basic Usage : ```python train.py data_directory```
  * Prints out current epoch, training loss, validation loss, and validation accuracy as the network trains
  * Options:
    * Set directory to save checkpoints: ```python train.py data_dir --save_dir save_directory```
    * Choose architecture (resnet, densenet121 or vgg16 available): ```python train.py data_dir --arch "vgg13"```
    * Set hyperparameters: ```python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20```
    * Use GPU for training: ```python train.py data_dir --gpu```

* Predict flower name from an image with ```predict.py``` along with the probability of that name. That is you'll pass in a single image ```/path/to/image``` and return the flower name and class probability
  * Basic usage: ```python predict.py /path/to/image checkpoint```
  * Options:
    * Return top **K** most likely classes:```python predict.py input checkpoint --top_k 3```
    * Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_to_name.json```
    * Use GPU for inference: ```python predict.py input checkpoint --gpu```


**Sample Training Command:**
```
python train.py  --arch "vgg13" --gpu --epochs 10 --hidden_units 512 --learning_rate 0.001
```

**Sample Prediction Command:**
```
python predict.py 'flowers/test/5/image_05159.jpg' --gpu --category_names cat_to_name.json --topk 5
```

## Authors

* **Daniel Jaensch**
* **Udacity** - *Final Project of the AI with Python Nanodegree*
