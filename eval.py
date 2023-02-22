import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

model = tf.keras.models.load_model("output_model")
[train_ds, test_ds], ds_info = tfds.load('cars196', split=['train', 'test'], shuffle_files=True, as_supervised=True,
                                         with_info=True)
labels = ds_info.features['label'].names
img_height = 224
img_width = 224
batch_size = 32
val_batches = 50

resize = tf.keras.layers.Resizing(img_height, img_width)
train_ds = train_ds.map(lambda img, lab: (resize(img), lab)).shuffle(1024)
val_ds = train_ds.take(val_batches).batch(batch_size)
train_ds = train_ds.skip(val_batches).batch(batch_size)
test_ds = test_ds.map(lambda img, lab: (resize(img), lab)).batch(batch_size)


def get_actual_predicted_labels(dataset):
    actual = [labels for _, labels in dataset.unbatch()]
    predicted = model.predict(dataset)

    actual = tf.stack(actual, axis=0)
    predicted = tf.concat(predicted, axis=0)
    predicted = tf.argmax(predicted, axis=1)

    return actual, predicted


actual, predicted = get_actual_predicted_labels(test_ds)


def calculate_classification_metrics(y_actual, y_pred, labels):
    """
      Calculate the precision and recall of a classification model using the ground truth and
      predicted values.

      Args:
        y_actual: Ground truth labels.
        y_pred: Predicted labels.
        labels: List of classification labels.

      Return:
        Precision and recall measures.
    """
    cm = tf.math.confusion_matrix(y_actual, y_pred)
    tp = np.diag(cm)  # Diagonal represents true positives
    precision = dict()
    recall = dict()
    for i in range(len(labels)):
        col = cm[:, i]
        fp = np.sum(col) - tp[i]  # Sum of column minus true positive is false negative

        row = cm[i, :]
        fn = np.sum(row) - tp[i]  # Sum of row minus true positive, is false negative

        precision[labels[i]] = tp[i] / (tp[i] + fp)  # Precision

        recall[labels[i]] = tp[i] / (tp[i] + fn)  # Recall

    return precision, recall


precision, recall = calculate_classification_metrics(actual, predicted, labels)
breakpoint()
