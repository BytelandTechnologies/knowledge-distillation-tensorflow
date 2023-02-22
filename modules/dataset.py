import tensorflow as tf
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt


class Dataset:
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    AUTOTUNE = tf.data.AUTOTUNE

    def __init__(self, plot_samples=True):
        (train_ds, val_ds, test_ds), self.__ds_info = tfds.load("stanford_online_products",
                                                                with_info=True,
                                                                shuffle_files=True,
                                                                split=['train[:75%]', 'train[75%:]', 'test'])
        self.__data_augmentation = self.get_daug_model()
        self.__train_ds = self.__prepare(train_ds, shuffle=True, augment=True)
        self.__val_ds = self.__prepare(val_ds)
        self.__test_ds = self.__prepare(test_ds)

        if plot_samples:
            self.plot_samples(self.__train_ds, title="Training samples")
            self.plot_samples(self.__val_ds, title="Validation samples")

    def __prepare(self, ds, shuffle=False, augment=False):
        ds = ds.map(lambda x: (self.__prep_img(x['image']), x['super_class_id']), num_parallel_calls=self.AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(1024)

        if augment:
            ds = ds.map(lambda x, y: (self.__data_augmentation(x, training=True), y), num_parallel_calls=self.AUTOTUNE)

        return ds.batch(self.BATCH_SIZE).prefetch(buffer_size=self.AUTOTUNE)

    def __prep_img(self, image):
        image = tf.image.resize(image, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        return image

    @staticmethod
    def get_daug_model():
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(.1),
            tf.keras.layers.RandomContrast(.2),
        ])

    @staticmethod
    def plot_samples(ds, title="Dataset samples"):
        fig = plt.figure(figsize=(10, 10))
        ds = ds.take(1).unbatch().map(lambda x, y: tf.cast(x, dtype=tf.int32)).take(32)
        for i, image in enumerate(ds.as_numpy_iterator()):
            plt.subplot(4, 8, i + 1)
            plt.imshow(image)
            plt.axis("off")
        fig.suptitle(title)
        plt.show()

    @property
    def ds_info(self):
        return self.__ds_info

    @property
    def train_ds(self):
        return self.__train_ds

    @property
    def val_ds(self):
        return self.__val_ds

    @property
    def test_ds(self):
        return self.__test_ds
