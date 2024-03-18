import tensorflow as tf

#import dataset
train_dataset = "./images/train"
test_dataset = "./images/test"
val_dataset = "./images/valid"

# Convert to numpy arrays and extract indices
train_idxs = list(train_dataset.as_numpy_iterator())
test_idxs = list(test_dataset.as_numpy_iterator())
val_idxs = list(val_dataset.as_numpy_iterator())

#creating DataFrames

train_df = tf.data.Dataset.from_tensor_slices(train_idxs)
test_df = tf.data.Dataset.from_tensor_slices(test_idxs)
val_df = tf.data.Dataset.from_tensor_slices(val_idxs)
