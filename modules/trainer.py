import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight


class Trainer:

    def __init__(self, num_classes):
        base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False, weights='imagenet')
        base_model.trainable = False

        self.__model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(64, activation='elu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-3, decay_steps=30e3)
        self.__model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])
        self.__model.summary()

    def train_model(self, train_ds, val_ds, epochs=5):
        labs_train = np.array([lab for _, lab in train_ds.unbatch().as_numpy_iterator()])
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labs_train), y=labs_train)
        class_weights = dict(enumerate(class_weights))
        return self.__model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weights)

    def evaluate_model(self, test_ds):
        self.__model.evaluate(test_ds)

    def save_model(self, filename):
        self.__model.save(filename)
