# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SimCLR prediction heads."""

from typing import Text, Optional

import tensorflow as tf

from official.vision.beta.projects.simclrRP.modeling.layers import nn_blocks

regularizers = tf.keras.regularizers
layers = tf.keras.layers
from tensorflow_addons.layers import InstanceNormalization as INorm
from tensorflow.keras.layers import Conv2D


class ProjectionHead(tf.keras.layers.Layer):
  """Projection head."""

  def __init__(
      self,
      num_proj_layers: int = 3,
      proj_output_dim: Optional[int] = None,
      ft_proj_idx: int = 0,
      kernel_initializer: Text = 'VarianceScaling',
      kernel_regularizer: Optional[regularizers.Regularizer] = None,
      bias_regularizer: Optional[regularizers.Regularizer] = None,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      **kwargs):
    """The projection head used during pretraining of SimCLR.

    Args:
      num_proj_layers: `int` number of Dense layers used.
      proj_output_dim: `int` output dimension of projection head, i.e., output
        dimension of the final layer.
      ft_proj_idx: `int` index of layer to use during fine-tuning. 0 means no
        projection head during fine tuning, -1 means the final layer.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
        Default to None.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      **kwargs: keyword arguments to be passed.
    """
    super(ProjectionHead, self).__init__(**kwargs)

    assert proj_output_dim is not None or num_proj_layers == 0
    assert ft_proj_idx <= num_proj_layers, (num_proj_layers, ft_proj_idx)

    self._proj_output_dim = proj_output_dim
    self._num_proj_layers = num_proj_layers
    self._ft_proj_idx = ft_proj_idx
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._layers = []

  def get_config(self):
    config = {
        'proj_output_dim': self._proj_output_dim,
        'num_proj_layers': self._num_proj_layers,
        'ft_proj_idx': self._ft_proj_idx,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    base_config = super(ProjectionHead, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    self._layers = []
    if self._num_proj_layers > 0:
      intermediate_dim = int(input_shape[-1])
      for j in range(self._num_proj_layers):
        if j != self._num_proj_layers - 1:
          # for the middle layers, use bias and relu for the output.
          layer = nn_blocks.DenseBN(
              output_dim=intermediate_dim,
              use_bias=True,
              use_normalization=True,
              activation='relu',
              kernel_initializer=self._kernel_initializer,
              kernel_regularizer=self._kernel_regularizer,
              bias_regularizer=self._bias_regularizer,
              use_sync_bn=self._use_sync_bn,
              norm_momentum=self._norm_momentum,
              norm_epsilon=self._norm_epsilon,
              name='nl_%d' % j)
        else:
          # for the final layer, neither bias nor relu is used.
          layer = nn_blocks.DenseBN(
              output_dim=self._proj_output_dim,
              use_bias=False,
              use_normalization=True,
              activation=None,
              kernel_regularizer=self._kernel_regularizer,
              kernel_initializer=self._kernel_initializer,
              use_sync_bn=self._use_sync_bn,
              norm_momentum=self._norm_momentum,
              norm_epsilon=self._norm_epsilon,
              name='nl_%d' % j)
        self._layers.append(layer)
    super(ProjectionHead, self).build(input_shape)

  def call(self, inputs, training=None):
    hiddens_list = [tf.identity(inputs, 'proj_head_input')]

    if self._num_proj_layers == 0:
      proj_head_output = inputs
      proj_finetune_output = inputs
    else:
      for j in range(self._num_proj_layers):
        hiddens = self._layers[j](hiddens_list[-1], training)
        hiddens_list.append(hiddens)
      proj_head_output = tf.identity(
          hiddens_list[-1], 'proj_head_output')
      proj_finetune_output = tf.identity(
          hiddens_list[self._ft_proj_idx], 'proj_finetune_output')

    # The first element is the output of the projection head.
    # The second element is the input of the finetune head.
    return proj_head_output, proj_finetune_output


class ClassificationHead(tf.keras.layers.Layer):
  """Classification Head."""

  def __init__(
      self,
      num_classes: int,
      kernel_initializer: Text = 'random_uniform',
      kernel_regularizer: Optional[regularizers.Regularizer] = None,
      bias_regularizer: Optional[regularizers.Regularizer] = None,
      name: Text = 'head_supervised',
      **kwargs):
    """The classification head used during pretraining or fine tuning.

    Args:
      num_classes: `int` size of the output dimension or number of classes
        for classification task.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
        Default to None.
      name: `str`, name of the layer.
      **kwargs: keyword arguments to be passed.
    """
    super(ClassificationHead, self).__init__(name=name, **kwargs)
    self._num_classes = num_classes
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._name = name

  def get_config(self):
    config = {
        'num_classes': self._num_classes,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    }
    base_config = super(ClassificationHead, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    self._dense0 = layers.Dense(
        units=self._num_classes,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=None)
    super(ClassificationHead, self).build(input_shape)

  def call(self, inputs, training=None):
    inputs = self._dense0(inputs)
    return inputs

import math
class UpConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3,3), padding='same',name='UPConv'):
        # super(UpConvBlock, self).__init__(name=f"UpConvBlock_{UpConvBlock.count}")
        super(UpConvBlock, self).__init__(name=name)
        self.forward = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters, kernel_size, 1, padding),])
        self.forward.add(layers.LeakyReLU(0.2))
        self.forward.add(tf.keras.layers.UpSampling2D((2,2)))
        # self.forward = tf.keras.models.Sequential([tf.keras.layers.Conv2D(filters, kernel_size, 1, padding,activation='relu'),])
        # self.forward.add( tf.keras.layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding='same',activation='relu'))
        
    def call(self, inputs):
        return self.forward(inputs)

class LgtDecoder(tf.keras.layers.Layer):
    def __init__(self, z_dim=1024, tgtimage=32, name='decoder'):
        super(LgtDecoder, self).__init__(name=name)
        self.steps=int(math.log2(tgtimage))
        self.d1=tf.keras.layers.Dense(z_dim*4, activation='relu')
        self.d2=tf.keras.layers.Dense(z_dim, activation='relu')
        self.us=[]
        filt_cnt=2048 #512
        for ii in range(self.steps):
          self.us.append(UpConvBlock(filters=filt_cnt, kernel_size=(3,3)))  #,name=self.name+ 'up' +str(ii),name=self.name+ 'up' +str(ii)
          filt_cnt = filt_cnt/2  
        # self.c1=tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=1, padding='same', activation='sigmoid')
        self.c2=tf.keras.layers.Conv2D(filters=filt_cnt/2, kernel_size=(3,3), strides=1, padding='same', activation='relu')
        self.c3=tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=1, padding='same', activation='sigmoid')
        self._config_dict = {'z_dim': z_dim, 'tgtimage': tgtimage}
        self.layBF=None

    def get_config(self) :
      """Gets the config of this model."""
      return self._config_dict

    def call(self, inputs):
      x=self.d1(inputs)
      x=self.d2(x)
      for ii in range(self.steps):
        x=self.us[ii](x)
      # self.layBF=x
      x=self.c2(x)
      x=self.c3(x)
      # ylist= tf.split(self.layBF, 2, 0)
      # y=tf.concat(ylist, axis=-1)
      y=None #self.c2(self.layBF)
      # y=self.c3(y)
      return [x,x]

class EnConvBlock(tf.keras.layers.Layer):
    def __init__(self, depth,filters,name='EnConv'):
        super(EnConvBlock, self).__init__(name=name)
        self.forward = tf.keras.models.Sequential([Conv2D(filters, (4,4),strides= 2,padding='same'),])
        # self.forward.add( INorm()) 
        self.forward.add( layers.Activation('relu')) 

    def call(self, inputs):
        return self.forward(inputs)

class LgtEncoder(tf.keras.layers.Layer):
    def __init__(self,  name='encoder'):
        super(LgtEncoder, self).__init__(name=name)
        self.steps=5
        nout = [ 0,  2048,1028, 512, 256, 128, 3]
        self.encodes=[]
        for d in range(self.steps, 0, -1):
            tmpEncode=EnConvBlock(d, nout[d])
            self.encodes.append(tmpEncode)
        self._config_dict = {'step': self.steps}

    def get_config(self) :
      """Gets the config of this model."""
      return self._config_dict

    def call(self, inputs):
      res={}
      x=inputs
      for ii in range(self.steps):
        x=self.encodes[ii](x)
        res[ii]=x
      return res

# class LgtDecoderX(tf.keras.layers.Layer):
#     def __init__(self, z_dim=1024, tgtimage=32, name='decoder'):
#         super(LgtDecoderX, self).__init__(name=name)
#         self.steps=int(math.log2(tgtimage))
#         self.d1=tf.keras.layers.Dense(z_dim*4, activation='relu')
#         self.d2=tf.keras.layers.Dense(z_dim, activation='relu')
#         self.us=[]
#         filt_cnt=2048 #512
#         for ii in range(self.steps):
#           self.us.append(UpConvBlock(filters=filt_cnt, kernel_size=(3,3)))  #,name=self.name+ 'up' +str(ii),name=self.name+ 'up' +str(ii)
#           filt_cnt = filt_cnt/2  
#         # self.c1=tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=1, padding='same', activation='sigmoid')
#         self.c2=tf.keras.layers.Conv2D(filters=filt_cnt/2, kernel_size=(3,3), strides=1, padding='same', activation='relu')
#         self.c3=tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=1, padding='same', activation='sigmoid')
#         self._config_dict = {'z_dim': z_dim, 'tgtimage': tgtimage}
#         self.layBF=None

#     def get_config(self) :
#       """Gets the config of this model."""
#       return self._config_dict

#     def call(self, inputs):
#       x=self.d1(inputs)
#       x=self.d2(x)
#       for ii in range(self.steps):
#         x=self.us[ii](x)
#       # self.layBF=x
#       x=self.c2(x)
#       x=self.c3(x)
#       # ylist= tf.split(self.layBF, 2, 0)
#       # y=tf.concat(ylist, axis=-1)
#       y=None #self.c2(self.layBF)
#       # y=self.c3(y)
#       return [x,x]