import numpy as np 
import cv2
import random
import datetime
import traceback
import tensorflow as tf 
import tqdm
import random
import pathlib
import itertools
import collections

import cv2
import einops
import numpy as np
 
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers




def format_frames(frame, output_size):
  """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
  """
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame, *output_size)
  return frame


def frames_from_video_file(video, n_frames, output_size = (224,224), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
  """
  # Read each video frame by frame
  result = []
  #src = cv2.VideoCapture(str(video_path))  

  src = video

  video_length = len(src)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)
 
  # ret is a boolean indicating whether read was successful, frame is the image itself
 

  for _ in range(n_frames - 1):
    frame = video[start]
    frame = format_frames(frame, output_size)
    result.append(frame)
    start += frame_step
    if start >= video_length:
        break
 
  result = np.array(result)[..., [2, 1, 0]]

  return result






#--------------------------------------- model -----------------------------------------#
class ResidualMain(keras.layers.Layer):
  """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
  """
  def __init__(self, filters, kernel_size):
    super().__init__()
    self.seq = keras.Sequential([
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization(),
        layers.ReLU(),
        Conv2Plus1D(filters=filters, 
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)
  
  def get_config(self):

    config = super().get_config().copy()
    config.update({
        'seq': self.seq
    })
    return config

class Conv2Plus1D(keras.layers.Layer):
  def __init__(self, filters, kernel_size, padding):
    """
      A sequence of convolutional layers that first apply the convolution operation over the
      spatial dimensions, and then the temporal dimension. 
    """
    super().__init__()
    self.seq = keras.Sequential([  
        # Spatial decomposition
        layers.Conv3D(filters=filters,
                      kernel_size=(1, kernel_size[1], kernel_size[2]),
                      padding=padding),
        # Temporal decomposition
        layers.Conv3D(filters=filters, 
                      kernel_size=(kernel_size[0], 1, 1),
                      padding=padding)
        ])

  def call(self, x):
    return self.seq(x)
  def get_config(self):

    config = super().get_config().copy()
    config.update({
        'seq': self.seq
    })
    return config
  
class Project(keras.layers.Layer):
  """
    Project certain dimensions of the tensor as the data is passed through different 
    sized filters and downsampled. 
  """
  def __init__(self, units):
    super().__init__()
    self.seq = keras.Sequential([
        layers.Dense(units),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)
  

  def get_config(self):

    config = super().get_config().copy()
    config.update({
        'seq': self.seq
    })
    return config
def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = ResidualMain(filters, 
                     kernel_size)(input)

  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])


class ResizeVideo(keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = layers.Resizing(self.height, self.width)

  def call(self, video):
    """
      Use the einops library to resize the tensor.  

      Args:
        video: Tensor representation of the video, in the form of a set of frames.

      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for height, 
    # w stands for width, and c stands for the number of channels.
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return videos
  def get_config(self):

    config = super().get_config().copy()
    config.update({
        'height': self.height,
        'width': self.width,
        'resizing_layer': self.resizing_layer
    })
    return config






from tensorflow.keras.layers import Dropout
 
HEIGHT = 224
WIDTH = 224
input_shape = (None, 10, HEIGHT, WIDTH, 3)
input = layers.Input(shape=(input_shape[1:]))
x = input

x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
x = Dropout(0.1)(x)
x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)

# Block 1
x = add_residual_block(x, 16, (3, 3, 3))
x = Dropout(0.1)(x)
x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)

# Block 2
x = add_residual_block(x, 32, (3, 3, 3))
x = Dropout(0.1)(x)
x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

# Block 3
x = add_residual_block(x, 64, (3, 3, 3))
x = Dropout(0.1)(x)
x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

# Block 4
x = add_residual_block(x, 128, (3, 3, 3))
x = Dropout(0.1)(x)
x = ResizeVideo(HEIGHT // 32, WIDTH // 32)(x) 


x = layers.AveragePooling3D((10,1,1))(x)
x = layers.Reshape((x.shape[1]*x.shape[2]*x.shape[3],-1))(x)
x = layers.LSTM(128,return_sequences=True)(x)
x = layers.Flatten()(x)
x = layers.Dense(512)(x)
x = Dropout(0.1)(x)
x = layers.Dense(256)(x)

x = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(input, x)
model.summary()


model.load_weights("weights.h5")


n_frames = 10

time_init = datetime.now().timestamp()

#Extraemos 5 segundos de datos
duration = 5 

frames = list()
while 1: 

    try:
        vid = cv2.VideoCapture(r"rtsp://admin:123456789A+@192.168.1.4:554/streaming/channels/1")
        _, frame = vid.read()
    except:
        print(f"[ERROR] Error al obtener flujo de video. Error: {traceback.format_exc()}")
    else:
        frames.append(frame)

    if datetime.now().timestamp() - time_init > duration:
        break


result = frames_from_video_file(frames,n_frames)


predict = model.predict(result)

print("result",predict)
