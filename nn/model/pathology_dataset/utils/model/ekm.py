import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
import tensorflow.keras.layers as L
from tensorflow.keras.optimizers import Adam, RMSprop

def ekm(drop):
    METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
        tf.keras.metrics.AUC(name="categorical_auc", multi_label=True)
    ]

    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 200x200 with 3 bytes color
        tf.keras.layers.Conv2D(16, (3,3), activation="relu", input_shape=(200,200,3)),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(32, (3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64,(3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Conv2D(64,(3,3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(drop),
        tf.keras.layers.Dense(4, activation="softmax")
    ])

    model.compile(RMSprop(lr=5e-4, momentum=0.1), loss = "categorical_crossentropy", metrics=METRICS)

    print(model.summary())

    return model