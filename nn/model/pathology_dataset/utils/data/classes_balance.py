import tensorflow as tf
import numpy as np
from imblearn.over_sampling import SMOTE  # Cqlass Balancing tool
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img,
    save_img,
)


def create_augmented_data(
    train_dir,
    train_generator,
    val_generator,
    augmentation,
    batch_size,
    train_size,
    val_size,
    seed,
    image_size,
    valid=True,
):
    """Perform class balancing with SMOTE and data augmentation with TF's ImageDataGenerator

    Parameters
    ----------
    train_dir:
        training directory
    train_generator:
        training generator
    val_generator:
        validating generator
    augmentation:
        augmentation methods
    batch_size:
        size of training batch
    valid:
        true = output both train, validation data
        false = output train data only
    train_size:
        global variable training size
    val_size:
        global variable validating size
    seed:
        global variable constant random generator seed
    image_size:
        global variable image size

    Return
    ----------
    if:
        out_train_flow:
        out_val_flow:
        y_val:
        y_train:
        total_train:
    else:
        out_train_flow:
        total_train:
        xforpca:
        yforpca:
        xforpca1:
        yforpca1:
    """
    if valid:
        # Load data into tensorflow dataset
        print("Loading Data...")
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="training",
            seed=1337,
            image_size=image_size,
            batch_size=train_size,
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="validation",
            seed=1337,
            image_size=image_size,
            batch_size=val_size,
        )

        print("Augmenting train...")
        res = list(zip(*train_ds.unbatch().as_numpy_iterator()))
        x_train = np.array(res[0])
        print("x done")
        y_train = np.array(res[1])
        yforpca = y_train
        print(x_train.shape, y_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution before SMOTE = ", counts)
        x_train = np.array([image.flatten() for image in x_train])
        xforpca = x_train
        print("Flattened")

        smote_train = SMOTE(
            sampling_strategy="all", random_state=420, k_neighbors=10, n_jobs=4
        )  # svmsmote goes out of memory in all configs
        x_train, y_train = smote_train.fit_resample(x_train, y_train)
        x_train = np.reshape(x_train, (-1, 200, 200, 3))
        total_train = len(x_train)
        print("Total_train after smote = ", x_train.shape)
        yforpca1 = y_train
        xforpca1 = x_train
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution after SMOTE = ", counts)
        y_train_cat = tf.keras.utils.to_categorical(
            y_train, num_classes=4, dtype="float32"
        )

        train_generator.fit(x_train, seed=seed)
        aug_train_images, aug_train_labels = train_generator.flow(
            x=x_train, y=y_train_cat, shuffle=False, batch_size=total_train, seed=seed
        ).next()
        aug_train_images = np.array(aug_train_images)
        aug_train_labels = np.array(aug_train_labels)
        # Save memory
        del x_train
        # del y_train
        del train_ds

        out_train_datagen = ImageDataGenerator()
        out_train_datagen.fit(aug_train_images)
        out_train_flow = out_train_datagen.flow(
            aug_train_images, aug_train_labels, batch_size=batch_size, shuffle=False
        )

        del aug_train_images
        del aug_train_labels

        print("Train augmented, augmenting val...")
        # i = 0
        res = list(zip(*val_ds.unbatch().as_numpy_iterator()))
        x_val = np.array(res[0])
        y_val = np.array(res[1])
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=4, dtype="float32")
        print(x_val.shape, y_val.shape, y_val_cat.shape)

        val_generator.fit(x_val)
        aug_val_images, aug_val_labels = val_generator.flow(
            x=x_val, y=y_val_cat, shuffle=False, batch_size=val_size, seed=seed
        ).next()
        aug_val_images = np.array(aug_val_images)
        aug_val_labels = np.array(aug_val_labels)

        del x_val
        del val_ds

        out_val_datagen = ImageDataGenerator()
        out_val_datagen.fit(aug_val_images)
        out_val_flow = out_val_datagen.flow(
            aug_val_images, aug_val_labels, batch_size=val_size, shuffle=False
        )

        del aug_val_images
        del aug_val_labels
        del res

        print("Returning")
        return (out_train_flow, out_val_flow, y_val, y_train, total_train)

    ##################################################################################
    else:
        # validation is not provided
        print("Loading data...")
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            seed=1337,
            image_size=image_size,
            batch_size=train_size,
        )

        print("Augmenting train...")
        res = list(zip(*train_ds.unbatch().as_numpy_iterator()))
        x_train = np.array(res[0])
        y_train = np.array(res[1])
        print(x_train.shape, y_train.shape)
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution before smote = ", counts)
        x_train = np.array([image.flatten() for image in x_train])
        print("Flattened")
        yforpca = y_train
        xforpca = x_train
        smote_train = SMOTE(
            sampling_strategy="all", random_state=420, k_neighbors=10, n_jobs=4
        )
        x_train, y_train = smote_train.fit_resample(x_train, y_train)
        x_train = np.reshape(x_train, (-1, 200, 200, 3))
        yforpca1 = y_train
        xforpca1 = x_train
        unique, counts = np.unique(y_train, return_counts=True)
        print("Class distribution after smote = ", counts)
        total_train = len(x_train)
        print("Total_train after smote = ", x_train.shape)

        y_train_cat = tf.keras.utils.to_categorical(
            y_train, num_classes=4, dtype="float32"
        )

        train_generator.fit(x_train, seed=seed)
        aug_train_images, aug_train_labels = train_generator.flow(
            x=x_train, y=y_train_cat, shuffle=False, batch_size=total_train, seed=seed
        ).next()
        aug_train_images = np.array(aug_train_images)
        aug_train_labels = np.array(aug_train_labels)

        del x_train
        del y_train
        del train_ds

        out_train_datagen = ImageDataGenerator()
        out_train_datagen.fit(aug_train_images)
        out_train_flow = out_train_datagen.flow(
            aug_train_images, aug_train_labels, batch_size=batch_size, shuffle=False
        )

        del aug_train_images
        del aug_train_labels

        return (out_train_flow, total_train, xforpca, yforpca, xforpca1, yforpca1)


def get_augmented_test(test_dir, test_generator, outdir, test_size, image_size, seed):
    """Test set is preprocessed just as validation set, in order to give the model the same feature distribution

    Parameters
    ----------
    test_dir:
        testing directory
    test_generator:
        test generator
    outdir:
        output directory
    test_size:
        testing size
    image_size:
        image size

    Return
    ----------
    test_imgs:
        testing image
    """
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        outdir + "test_image",
        labels="inferred",
        label_mode="int",
        class_names=None,
        color_mode="rgb",
        batch_size=test_size,
        image_size=image_size,
        shuffle=False,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )

    x_test = np.array([array for array, label in test_ds.unbatch().as_numpy_iterator()])
    test_generator.fit(x_test, seed=seed)
    test_flow = test_generator.flow(
        x=x_test, y=None, batch_size=test_size, shuffle=False, seed=seed
    )

    test_imgs = test_flow.next()

    del test_ds
    del x_test
    del test_generator

    return test_imgs
