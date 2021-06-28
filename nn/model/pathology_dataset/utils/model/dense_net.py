import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
import tensorflow.keras.layers as L

def densenet121():
    METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
        tf.keras.metrics.AUC(name="categorical_auc", multi_label=True),
    ]

    model = tf.keras.Sequential([DenseNet121(input_shape=(200, 200, 3),
                                weights="imagenet",
                                include_top=False),
                                L.GlobalAveragePooling2D(),
                                L.Dense(4, activation="softmax")])
    
    model.compile(optimizer="adam",
                loss = "categorical_crossentropy",
                metrics=METRICS)

    print(model.summary())
    return model