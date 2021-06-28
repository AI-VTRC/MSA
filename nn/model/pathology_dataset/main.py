# Basic utilities
import sys  # Enabler of operating system dependent functionality
import os  # Provides access to some variables & functions for the interpreter
import shutil
from shutil import copyfile  # Import module we'll need to import our custom module
import math  # Provides access to basic mathematical functions
import time  # Provides various time-related functions
import glob  # Pathnames management
from PIL import Image as pil_image
import itertools

# Data manipulation
import pandas as pd
import numpy as np
from numpy import expand_dims

# Data Visualization
import matplotlib  # Interface for creation of publication-quality plots and figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import seaborn as sns  # Matplotlib-based statistical data visualization interface

# Plotly
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

# Scikit-Learn
from sklearn.model_selection import (
    train_test_split,
)  # split arrays or matrices into random train and test subsets
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import (
    PCA,
    TruncatedSVD,
)  # Principal component analysis (PCA), dimensionality reduction using truncated SVD

# Tensorflow - Keras - TF version 2.5
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img,
    save_img,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import tensorflow.keras.layers as L
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.optimizers import Adam, RMSprop

# SMOTEq
from imblearn.over_sampling import SMOTE  # Cqlass Balancing tool

# Other classes
from utils.data.preprocess import train_test_split
from utils.data.classes_balance import create_augmented_data
from utils.data.classes_balance import get_augmented_test
from utils.data.svd import svd
from utils.data.pca import pca
from utils.training import train_val

#################################################################################################
# Data Exploration
OUTDIR = "../pathology_dataset/dataset/classes_split/"
INDIR = "../pathology_dataset/dataset/"

TESTDIR = INDIR + "working/test_images/"

TRAIN_LABELS_CSV = pd.read_csv(INDIR + "train.csv")
print(TRAIN_LABELS_CSV.head())
print("------------------------------------------")

EXAMPLE_SUBMISSION_CSV = pd.read_csv(INDIR + "sample_submission.csv")
print(EXAMPLE_SUBMISSION_CSV)
print("------------------------------------------")

TEST_CSV = pd.read_csv(INDIR + "test.csv")
TEST_PATH_CSV = pd.DataFrame(TEST_CSV["image_id"].apply(lambda x: TESTDIR + x + ".jpg"))
print(TEST_PATH_CSV.head())

#################################################################################################
"""Split training and testing dataset into healthy, multiple_diseases, rust, scab

NOTE: This script is run only once, if want to rerun, you have to delete dataset/classes_split/train and dataset/classes_split/test_image
"""

# Create csv files with images
train_val_healthy_csv = TRAIN_LABELS_CSV[TRAIN_LABELS_CSV["healthy"] == 1]
train_val_multiple_diseases_csv = TRAIN_LABELS_CSV[
    TRAIN_LABELS_CSV["multiple_diseases"] == 1
]
train_val_rust_csv = TRAIN_LABELS_CSV[TRAIN_LABELS_CSV["rust"] == 1]
train_val_scab_csv = TRAIN_LABELS_CSV[TRAIN_LABELS_CSV["scab"] == 1]

train_val_healthy_names = train_val_healthy_csv["image_id"].tolist()
train_val_multiple_diseases_names = train_val_multiple_diseases_csv["image_id"].tolist()
train_val_rust_names = train_val_rust_csv["image_id"].tolist()
train_val_scab_names = train_val_scab_csv["image_id"].tolist()

src_img_dir = INDIR + "working/train_val_images"
out_split_dir = OUTDIR + "train"  # ../pathology_dataset/dataset/classes_split/train

# split dir
train_healthy_dir = OUTDIR + "train/healthy"
train_multiple_diseases_dir = OUTDIR + "train/multiple_diseases"
train_rust_dir = OUTDIR + "train/rust"
train_scab_dir = OUTDIR + "train/scab"

test_dir = OUTDIR + "test_image/test"

# Create and fill image in to the directories
try:
    os.mkdir(out_split_dir)
    os.mkdir(train_healthy_dir)
    os.mkdir(train_multiple_diseases_dir)
    os.mkdir(train_rust_dir)
    os.mkdir(train_scab_dir)
    os.mkdir(OUTDIR + "test_image")
    os.mkdir(test_dir)

    for image in train_val_healthy_names:
        shutil.copy(src_img_dir + "/" + image + ".jpg", train_healthy_dir)
    for image in train_val_multiple_diseases_names:
        shutil.copy(src_img_dir + "/" + image + ".jpg", train_multiple_diseases_dir)
    for image in train_val_rust_names:
        shutil.copy(src_img_dir + "/" + image + ".jpg", train_rust_dir)
    for image in train_val_scab_names:
        shutil.copy(src_img_dir + "/" + image + ".jpg", train_scab_dir)
    for image in TEST_PATH_CSV["image_id"].tolist():
        shutil.copy(image, test_dir)

except FileExistsError as err:
    # except FileNotFoundError as err:
    print("Folder already exist")
    print(err)

# Check for possible errors
total = len([file for file in os.listdir(src_img_dir)])
train_healthy_total = len([file for file in os.listdir(train_healthy_dir)])
train_multiple_diseases_total = len(
    [file for file in os.listdir(train_multiple_diseases_dir)]
)
train_rust_total = len([file for file in os.listdir(train_rust_dir)])
train_scab_total = len([file for file in os.listdir(train_scab_dir)])

total = (
    train_healthy_total
    + train_multiple_diseases_total
    + train_rust_total
    + train_scab_total
)

train_size = math.ceil(total * 0.8)
val_size = total - train_size
test_size = TEST_CSV.size
image_size = (200, 200)
batch_size = 32
seed = 100
print(
    train_healthy_total,
    train_multiple_diseases_total,
    train_rust_total,
    train_scab_total,
)

#################################################################################################
def data_analysis():
    """
    It's important to stress the difference between data preprocessing and data augmentation:
        preprocessing refers to a well defined transformation applied to all data (in order to save memory, speed up execution, etc...)
        augmentation refers to a random modification applied to a random sample of the data to train a more rubust model.
    Therefore (selected) augmentation techniques are applied to train only, while validation and test sets receive just the preprocessing applied to train.
    """
    ###############################################
    # Data Generator:
    print("\nData Generator --------------------------------------------------")
    # TRAIN
    train_datagen = ImageDataGenerator(
        rotation_range=360,  # DATA AUGMENTATION
        # shear_range=.25,                  # DATA AUGMENTATION
        # zoom_range=.25,                   # DATA AUGMENTATION
        # width_shift_range=.25,            # DATA AUGMENTATION
        # height_shift_range=.25,           # DATA AUGMENTATION
        rescale=1.0 / 255,  # DATA MODIFICATION
        # brightness_range=[.5,1.5],        # DATA AUGMENTATION
        horizontal_flip=True,  # DATA AUGMENTATION
        # vertical_flip=True                # DATA AUGMENTATION
    )

    # VALIDATION
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # TEST
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # To train and validate
    train_flow_80, val_flow, y_val, y_train, total_train_80 = create_augmented_data(
        train_dir=out_split_dir,
        train_generator=train_datagen,
        val_generator=val_datagen,
        augmentation=5,
        batch_size=batch_size,
        train_size=train_size,
        val_size=val_size,
        seed=seed,
        image_size=image_size,
    )

    train_flow, total_train, x, y, xS, yS = create_augmented_data(
        train_dir=out_split_dir,
        train_generator=train_datagen,
        val_generator=val_datagen,
        augmentation=5,
        batch_size=32,
        train_size=train_size,
        val_size=val_size,
        seed=seed,
        image_size=image_size,
        valid=False,
    )

    ###############################################
    # SVD - Only need to run 1
    print("\nSVD --------------------------------------------------")
    # svd(x, y, xS, yS)

    ###############################################
    # PCA - Only need to run 1
    print("\nPCA --------------------------------------------------")
    # pca(x, y, xS, yS)

    return train_flow_80, val_flow, y_val, y_train, total_train_80, train_flow, total_train, x, y, xS, yS

def main():
    ###############################################
    # Split Train/Val and Test - Only need to run 1
    # train_test_split()

    ###############################################
    # SVD + PCA
    train_flow_80, val_flow, y_val, y_train, total_train_80, train_flow, total_train, x, y, xS, yS = data_analysis()

    ###############################################
    # Checkpoint callback: a learning rate modifier which saves the best model weights for both prediction and for stocasticity evaluation
    if not os.path.isdir("../pathology_dataset/saved_model/"):
        os.makedirs("../pathology_dataset/saved_model/")

    args = {
        "save_weights_only": True,
        "monitor": "val_categorical_auc",
        "mode": "max",
        "save_best_only": True,
        "verbose": 0
    }

    checkpoint_filepaths = [
        "../pathology_dataset/saved_model/drop0.hdf5",
        "../pathology_dataset/saved_model/drop1.hdf5",
        "../pathology_dataset/saved_model/drop2.hdf5",
        "../pathology_dataset/saved_model/drop3.hdf5",
        "../pathology_dataset/saved_model/drop4.hdf5",
        "../pathology_dataset/saved_model/drop5.hdf5",
        "../pathology_dataset/saved_model/drop6.hdf5",
        "../pathology_dataset/saved_model/drop7.hdf5",
        "../pathology_dataset/saved_model/drop8.hdf5",
        "../pathology_dataset/saved_model/drop9.hdf5",
    ]
    
    ###############################################
    # Model Training
    print("\nModel Training --------------------------------------------------")
    train_val(checkpoint_filepaths, args, batch_size, train_size, train_flow_80, val_flow, y_val, y_train, total_train_80, train_flow, total_train, x, y, xS, yS, TEST_CSV, test_dir)

if __name__ == "__main__":
    main()
