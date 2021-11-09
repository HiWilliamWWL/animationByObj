import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

batch_size = 2
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
      self.gate1 = layers.Dense(num_hid*4, activation='relu')
      self.gateD1 = layers.Dropout(0.1)
      self.gate2 = layers.Dense(num_hid*2, activation='relu')
      self.gateD2 = layers.Dropout(0.1)
      self.gate3 = layers.Dense(num_hid, activation='relu')
      self.gateD3 = layers.Dropout(0.1)
      self.gate4 = layers.Dense(num_hid)
      #self.gate5 = layers.Dense(num_experts, name="expert_weights", activation='softmax')  #63  54  54+63-3=114
      self.gate5 = layers.Dense(num_classes, name="expert_weights", activation='softmax')

      self.walkPortionDimensions = [num_input_dimension, num_hid*4, num_hid*2, num_hid*2, num_hid, num_classes]
      self.walk1 = mixWeightNN((self.walkPortionDimensions[0], self.walkPortionDimensions[1]), num_experts, "relu")
      self.walkD1 = layers.Dropout(0.1)
      self.walk2 = mixWeightNN((self.walkPortionDimensions[1], self.walkPortionDimensions[2]), num_experts, "relu")
      self.walkD2 = layers.Dropout(0.1)
      self.walk3 = mixWeightNN((self.walkPortionDimensions[2], self.walkPortionDimensions[3]), num_experts, "relu")
      self.walkD3 = layers.Dropout(0.1)
      self.walk4 = mixWeightNN((self.walkPortionDimensions[3], self.walkPortionDimensions[4]), num_experts, "relu")
      self.walk5 = mixWeightNN((self.walkPortionDimensions[4], self.walkPortionDimensions[5]), num_experts, None)

  def getExpertWeights(self, inputs):
    
    s1 = self.gate1(inputs)
    s1 = self.gateD1(s1)
    s1 = self.gate2(s1)
    s1 = self.gateD2(s1)
    s1 = self.gate3(s1)
    s1 = self.gateD3(s1)
    s1 = self.gate4(s1)
    s1 = self.gate5(s1)
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
    
    #source = inputs[0]  #obj
    #target = inputs[1]  #h
    
    result = None

    for i in range(0, self.target_maxlen ):

        #currentInput = tf.concat((inputs[:, i, :3], result[:, -1, :]), axis=-1)
        currentInput = inputs[:, i, :]

        flattenedInput = self.flattenInit(currentInput)

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
    x, y = batch
    x = tf.dtypes.cast(x, tf.float32)
    y = tf.dtypes.cast(y, tf.float32)  # (b, 85, 84)
    #x = tf.identity(y)
    #obj_center = self.getObjCenterInfo(batch_size, source)
    #y = tf.concat((target, obj_center), axis = -1)
    with tf.GradientTape() as tape:
      y_p = self(x, training=True)
      loss = self.compiled_loss(y_true=y[:, 1:, 3:], y_pred=y_p[:, 1:, :])
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    self.compiled_metrics.update_state(y_true=y[:, 1:, 3:], y_pred=y_p[:, 1:, :])
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, batch):
    x, y = batch
    x = tf.dtypes.cast(x, tf.float32)
    y = tf.dtypes.cast(y, tf.float32)
    #x = tf.identity(y)

    preds = self(x, training=False)
    self.compiled_metrics.update_state(y_true=y[:, 1:, 3:], y_pred=preds[:, 1:, :])
    return {m.name: m.result() for m in self.metrics}
  
  def generate(self, testX):
    # 1, 84
    flattenedInput = self.flattenInit(testX)

    #do the weight mixing
    expert_weights = self.getExpertWeights(flattenedInput)

    #current_result = expert_weights
    #current_result = self.getWalkInfo(flattenedInput, expert_weights)
    return expert_weights


#testModel = walkGenerateNet()