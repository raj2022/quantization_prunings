#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow_datasets as tfds
import time
import matplotlib.pyplot as plt


# In[2]:


def load_dataset(num_samples=100):
    ds_builder = tfds.builder("clic_edm_qq_pf", data_dir='../../../tensorflow_datasets/')
    dss = ds_builder.as_data_source("test")
    limited_dss = []

    for elem in dss:
        limited_dss.append(elem)
        if len(limited_dss) >= num_samples:
            break

    return limited_dss


# In[3]:


def warm_up_gpu(dss):
    for elem in dss:
        _ = tf.dtypes.cast(elem["X"], tf.float16)
        break


# In[4]:


def quantize_data(dss):
    original_outputs = []
    quantized_outputs = []

    for elem in dss:
        original_output = elem["X"]
        quantized_output = tf.dtypes.cast(original_output, tf.float16)

        original_outputs.extend(tf.reshape(original_output, [-1]))
        quantized_outputs.extend(tf.reshape(quantized_output, [-1]))

    return original_outputs, quantized_outputs


# In[5]:


def measure_inference_time(dss):
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

    


# In[6]:


def plot_histograms(original_outputs, quantized_outputs, quantization_type):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(original_outputs, bins=100, range=(0, 255), color='blue', alpha=0.7, label='Original')
    plt.title('Histogram Before Quantization')
    plt.xlim(-5,120)
#     plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(quantized_outputs, bins=100, range=(0, 255), color='orange', alpha=0.7, label='Quantized')
    plt.title(f'Histogram After {quantization_type} Quantization')  # Include quantization type in title
    plt.xlim(-5,120)
    #     plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()
    plt.show()


# In[25]:


if __name__ == "__main__":
    dss = load_dataset()
    warm_up_gpu(dss)
    original_outputs, quantized_outputs = quantize_data(dss)
    quantized_outputs = tf.stack(quantized_outputs).numpy()  # Convert to NumPy array
    measure_inference_time(dss)
    plot_histograms(original_outputs, quantized_outputs, 'FP16')


