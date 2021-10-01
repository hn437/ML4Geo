# ML4Geo

This repository was created for a machine learning seminar for geographers at University Heidelberg. It tries to create a raster which contains all building footprints based on a training with osm building data in the same region.

There are 4 important scripts. The CNN and Image generators are defined in model.py, the preprocessing.py turns an rgb raster into the training data, and unet.py defines training, predicting and evaluation.

The scripts are run through the main.py, at the top you can define some hyperparameters for preprocessing and training and at the botton you can choose if you want to run preprocessing, training or both.

It is **required** to have a folder named "data" at the highest level with a raster that should be classified, it should be called *ml4geo_raster.tif*, but that name can be changed in the script definitions.

The requirements on the environment in which the scripts are run can be found within the textfile *requirements_conda_win64.txt* (with those packages, the env is valid on windows 64-bit machines).
