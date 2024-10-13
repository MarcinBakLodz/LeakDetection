from torchvision import transforms
import numpy as np

def calculate_means(data):
    # Initialize variables to store the sum of values and the sum of squared differences for each channel
    sum_channels = np.zeros(4)
    sum_squared_diffs = np.zeros(4)

    # Iterate through the dataset and sum up values for each channel
    num_samples = 0
    for package in data:
        for sample in package:
          sum_channels += np.sum(sample, axis=(0, 1))
          num_samples += sample.shape[0] * sample.shape[1]

    # Calculate the mean of each channel
    mean_channels = sum_channels / num_samples

    num_samples = 0
    for package in data:
        for sample in package:
          sum_squared_diffs += np.sum((sample - mean_channels) ** 2, axis=(0, 1))
          num_samples += sample.shape[0] * sample.shape[1]

    # Calculate the variance of each channel
    variance_channels = sum_squared_diffs / num_samples

    # Calculate the standard deviation of each channel
    std_dev_channels = np.sqrt(variance_channels)
    return mean_channels, std_dev_channels

def normalize(data, means, stds):
  normalized_data = (data - means) / stds
  return normalized_data