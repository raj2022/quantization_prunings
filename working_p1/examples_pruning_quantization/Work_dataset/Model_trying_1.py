#importing necessary libraries
from matplotlib import pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import tensorflow_datasets as tfds

# Adding the path to the module
sys.path += ["../../../particleflow/mlpf/"]
from tfmodel.model_setup import make_model
from tfmodel.utils import parse_config

# Parsing the configuration
config, _ = parse_config("../../../particleflow/parameters/clic.yaml")

# Creating the model
model = make_model(config, tf.float32)
model.build((1, None, config["dataset"]["num_input_features"]))

model.summary()

# Loading the pre-trained weights
model.load_weights("weights-96-5.346523.hdf5", skip_mismatch=False, by_name=True)

# Reading the dataset
ds_builder = tfds.builder("clic_edm_qq_pf", data_dir="../../../tensorflow_datasets/")
dss = ds_builder.as_data_source("test")

# Defining a generator function
def yield_from_ds():
    for elem in dss:
        yield {"X": elem["X"], "ygen": elem["ygen"], "ycand": elem["ycand"]}

# Creating the dataset
output_signature = {k: tf.TensorSpec(shape=(None, v.shape[1])) for (k, v) in dss.dataset_info.features.items()}
tf_dataset = tf.data.Dataset.from_generator(yield_from_ds, output_signature=output_signature).take(100).padded_batch(batch_size=10)
data = list(tfds.as_numpy(tf_dataset))

Xs = [d["X"] for d in data]
ys = [d["ygen"] for d in data]

true_pts = []
pred_pts = []

# Processing the batches
for ibatch in range(len(Xs)):
    ret = model(Xs[ibatch])

    mask_true_particles = ys[ibatch][..., 0] != 0
    true_pt = ys[ibatch][mask_true_particles, 2]
    pred_pt = ret["pt"][mask_true_particles][..., 0].numpy()

    true_pts.append(true_pt)
    pred_pts.append(pred_pt)

# Concatenating the results
true_pt = np.concatenate(true_pts)
pred_pt = np.concatenate(pred_pts)

# Plotting the histogram
plt.hist(pred_pt / true_pt, bins=np.linspace(0, 3, 100))
plt.yscale("log")
plt.show()

