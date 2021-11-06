import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


batch_size = 4

class SlidingBaseFc(keras.Model):
  def __init__(
      self, num_hid=128, target_maxlen=85, num_classes=10):
      super().__init__()
      self.target_maxlen = target_maxlen