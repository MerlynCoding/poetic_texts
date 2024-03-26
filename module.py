import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Activation, Dense, LSTM

filepath=tf.keras.utils.get_file('shakespear.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
