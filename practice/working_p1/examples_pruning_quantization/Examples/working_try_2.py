import tensorflow as tf
import tensorflow_datasets as tfds
import time
import matplotlib.pyplot as plt

# Load the dataset
ds_builder = tfds.builder("clic_edm_qq_pf", data_dir='../../tensorflow_datasets/')
dss = ds_builder.as_data_source("test")

# Warm up the GPU (optional, but can help in some cases)
for elem in dss:
    _ = tf.dtypes.cast(elem["X"], tf.float16)
    break

# Lists to store original and quantized outputs
original_outputs = []
quantized_outputs = []

# Process 10 samples and store outputs
for i, elem in enumerate(dss):
    original_output = elem["X"]
    quantized_output = tf.dtypes.cast(original_output, tf.float16)

    original_outputs.append(original_output)
    quantized_outputs.append(quantized_output)

    print(f"Sample {i+1}")
    print(f"Original Output: {original_output}")
    print(f"Quantized Output: {quantized_output}")

    if i >= 9:  # Process 10 samples
        break

# Measure inference time
total_time = 0
num_samples = 0

for elem in dss:
    start_time = time.time()
    quantized_weights = tf.dtypes.cast(elem["X"], tf.float16)
    end_time = time.time()
    total_time += end_time - start_time
    num_samples += 1

    if num_samples >= 10:  # Process 10 samples
        break

average_inference_time = total_time / num_samples
print(f'Average Inference Time: {average_inference_time} seconds')

# Plotting the outputs
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(original_outputs[i], cmap='gray')
    plt.axis('off')
    plt.title(f'Original {i+1}')

    plt.subplot(2, 10, i+11)
    plt.imshow(quantized_outputs[i], cmap='gray')
    plt.axis('off')
    plt.title(f'Quantized {i+1}')

plt.tight_layout()
plt.show()

