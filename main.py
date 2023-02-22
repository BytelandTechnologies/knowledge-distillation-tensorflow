import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

from modules.dataset import Dataset

dataset = Dataset()

labels_train = np.array([lab for _, lab in dataset.train_ds.unbatch().as_numpy_iterator()])

base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False, weights='imagenet')
base_model.trainable = False

NUM_CLASSES = dataset.ds_info.features['super_class_id'].num_classes
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='elu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
class_weights = class_weight.compute_class_weight('balanced',
                                                  classes=np.unique(labels_train),
                                                  y=labels_train)
class_weights = dict(enumerate(class_weights))
lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=1e-3, decay_steps=30e3)
model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(dataset.train_ds, validation_data=dataset.val_ds, epochs=5, class_weight=class_weights)
model.save("output_model")

model.evaluate(dataset.test_ds)
print("Foi!")
