
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from baseEnDeCoder import BaseEncoderDecoderLayout


batch_size = 4
'''
    startingPose Generate: Obj85->H1
    Generative: Obj2, H1 -> H1
'''
class mixWeightNN(layers.Layer):
  def __init__(self, weightShape, num_experts = 3, activationFunction='relu'):
    super().__init__()
    input_d , output_d = weightShape
    self.thisBias = tf.zeros([output_d], tf.float32)
    self.all_weights = tf.random.normal([num_experts, input_d, output_d])
    self.activationLayer = None
    if activationFunction is not None:
      self.activationLayer = tf.keras.layers.Activation(activationFunction)
  

  def call(self, x, expert_weights):
    current_weight = tf.einsum("be,emn->bmn", expert_weights, self.all_weights)
    result = tf.einsum("bmn, bm -> bn", current_weight, x) + self.thisBias
    if self.activationLayer is not None:
      result = self.activationLayer(result)
    return result



class walkGenerateNet(keras.Model):
  def __init__(
      self, num_hid=128, target_maxlen=85, num_classes=27, num_experts = 3, num_input_dimension = 20):
      super().__init__()
      self.expertWeights = [[] for i in range(num_experts)]

      self.target_maxlen = target_maxlen
      
      self.flattenInit = layers.Flatten()

      self.objProcess = BaseEncoderDecoderLayout(num_hid, dim_output=num_hid)
      self.expertWeightNet = BaseEncoderDecoderLayout(num_hid, dim_output=num_classes)

      self.walkPortionDimensions = [num_input_dimension, num_hid*4, num_hid*2, num_hid*2, num_hid, num_classes]
      self.walk1 = mixWeightNN((self.walkPortionDimensions[0], self.walkPortionDimensions[1]), num_experts, "elu")
      self.walkD1 = layers.Dropout(0.1)
      self.walk2 = mixWeightNN((self.walkPortionDimensions[1], self.walkPortionDimensions[2]), num_experts, "elu")
      self.walkD2 = layers.Dropout(0.1)
      self.walk3 = mixWeightNN((self.walkPortionDimensions[2], self.walkPortionDimensions[3]), num_experts, "elu")
      self.walkD3 = layers.Dropout(0.1)
      self.walk4 = mixWeightNN((self.walkPortionDimensions[3], self.walkPortionDimensions[4]), num_experts, "elu")
      self.walk5 = mixWeightNN((self.walkPortionDimensions[4], self.walkPortionDimensions[5]), num_experts, None)

  def getExpertWeights(self, inputs):
    
    s1 = self.expertWeightNet(inputs)
    return s1
  
  def getWalkInfo(self, inputs, expert_weights):
    s2 = self.walk1(inputs, expert_weights)
    s2 = self.walkD1(s2)
    s2 = self.walk2(s2, expert_weights)
    s2 = self.walkD2(s2)
    s2 = self.walk3(s2, expert_weights)
    s2 = self.walkD3(s2)
    s2 = self.walk4(s2, expert_weights)
    s2 = self.walk5(s2, expert_weights)
    return s2

  def getObjCenterInfo(self, batch_size, source):
    obj_pos = tf.reshape(source, (batch_size, self.target_maxlen, 12,3))
    obj_center = tf.reduce_mean(obj_pos, axis=-2)
    return obj_center[:, :, :]
  

  def call(self, inputs):
    print("in call")
    o, x = inputs
    result = None
    
    objInfo = self.objProcess(o)
    
    for i in range(0, self.target_maxlen - 1):

        #currentInput = tf.concat((inputs[:, i, :3], result[:, -1, :]), axis=-1)
        currentInput = x[:, i, :]
        
        #'''
        if result is not None:
          currentInput = tf.concat((tf.expand_dims(result[:, -1, 0], axis=1), currentInput[:, 1:]), axis=-1)
        #'''
        
        flattenedInput = tf.concat((currentInput, objInfo), axis = -1)

        #do the weight mixing
        expert_weights = self.getExpertWeights(flattenedInput)

        current_result = expert_weights
        #current_result = self.getWalkInfo(flattenedInput, expert_weights)

        current_result = tf.expand_dims(current_result, axis=1)
        if result is None:
          result = current_result
        else:
          result = tf.concat([result, current_result], axis = 1)
    
    return result
    
  
  def train_step(self, batch):
    o, x, y = batch
    print(o.shape)
    print(x.shape)
    print(y.shape)
    o = tf.dtypes.cast(o, tf.float32)
    x = tf.dtypes.cast(x, tf.float32)
    y = tf.dtypes.cast(y, tf.float32)  # (b, 85, 84)
    #x = tf.identity(y)
    #obj_center = self.getObjCenterInfo(batch_size, source)
    #y = tf.concat((target, obj_center), axis = -1)
    with tf.GradientTape() as tape:
      y_p = self([o, x], training=True)
      loss = self.compiled_loss(y, y_p)
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    self.compiled_metrics.update_state(y, y_p)
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, batch):
    o, x, y = batch
    o = tf.dtypes.cast(o, tf.float32)
    x = tf.dtypes.cast(x, tf.float32)
    y = tf.dtypes.cast(y, tf.float32)  # (b, 85, 84)

    preds = self([o, x], training=False)
    #self.compiled_metrics.update_state(y_true=y[:, 1:, 3:], y_pred=preds[:, 1:, :])
    self.compiled_metrics.update_state(y, preds)
    return {m.name: m.result() for m in self.metrics}
  
  def generate(self, testX):
    # 1, 84, 36
    # 1, 84, N
    o, x = testX
    
    preds = self([o, x], training=False)

    #current_result = expert_weights
    #current_result = self.getWalkInfo(flattenedInput, expert_weights)
    return preds


#testModel = walkGenerateNet()