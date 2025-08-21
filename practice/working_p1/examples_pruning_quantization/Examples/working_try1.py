# Trying to work on the file on which the jupyter was crashing


import tensorflow as tf
import tensorflow_datasets as tfds
import time


ds_builder = tfds.builder("clic_edm_qq_pf", data_dir='../../tensorflow_datasets/')
dss = ds_builder.as_data_source("test")


print(dss)

#for elem in dss:
#    print({"X": elem["X"], "ygen": elem["ygen"], "ycand": elem["ycand"]})

# Applyting with 10 datapoints and checking the output
for elem in dss.take(10):
#    print({"X": elem["X"] })
    _ = tf.dtypes.cast(elem["X"], tf.float16)
    print(quantized_weights)
# Applying the Float 16 Quantization on the above "X" Features



# Measuring the inference time
total_time = 0
num_samples= 0 

for elem in dss:
    start_time = time.time()
    quantized_weights = tf.dtypes.cast(elem["X"], tf.float16)
    end_time = time.time()
    total_time += end_time -  start_time
    num_samples += 1
#    print(quantized_weights)


average_inference_time = total_time / num_samples
print(f'Average Inference Time: {average_inference_time} seconds')
