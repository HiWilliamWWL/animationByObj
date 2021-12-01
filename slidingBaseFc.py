import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

batch_size = 4

'''
    startingPose Generate: Obj85->H1
    Generative: Obj2, H1 -> H1
'''
class SlidingBaseFc(keras.Model):
  def __init__(
      self, num_hid=128, target_maxlen=85, num_classes=10):
      super().__init__()
      self.target_maxlen = target_maxlen

      self.flattenObj = layers.Flatten()
      self.initalPosStart1 = layers.Dense(num_hid*4, activation='relu')
      self.initalPosStart10 = layers.Dropout(0.1)
      self.initalPosStart2 = layers.Dense(num_hid*2, activation='relu')
      self.initalPosStart20 = layers.Dropout(0.1)
      self.initalPosStart25 = layers.Dense(num_hid, activation='relu')
      self.initalPosStart20 = layers.Dropout(0.1)
      self.initalPosStart25 = layers.Dense(num_hid)
      self.initalPosStart3 = layers.Dense(num_classes, name="intial_human")  #63  54  54+63-3=114

      self.processObjHumanFlatten = layers.Flatten()
      self.oh1 = layers.Dense(num_hid*4, activation='relu')
      self.oh1d = layers.Dropout(0.1)
      self.oh12 = layers.Dense(num_hid*4, activation='relu')
      self.oh12d = layers.Dropout(0.1)
      self.oh2 = layers.Dense(num_hid*2, activation='relu')
      self.oh2d = layers.Dropout(0.1)
      self.oh22 = layers.Dense(num_hid*2, activation='relu')
      self.oh22d = layers.Dropout(0.1)
      self.oh3 = layers.Dense(num_hid, activation='relu')
      self.oh3d = layers.Dropout(0.1)
      self.oh4 = layers.Dense(num_hid)
      self.oh4d = layers.Dense(num_classes, name="intial_human")  #63  54  54+63-3=114


  def getInitialPos(self, inputs):
    s1 = self.flattenObj(inputs)
    s1 = self.initalPosStart1(s1)
    s1 = self.initalPosStart10(s1)
    s1 = self.initalPosStart2(s1)
    s1 = self.initalPosStart20(s1)
    s1 = self.initalPosStart25(s1)
    s1 = self.initalPosStart3(s1)
    return s1
  
  def getFollowingPos(self, inputs):
    s2 = self.processObjHumanFlatten(inputs)
    s2 = self.oh1(s2)
    s2 = self.oh1d(s2)
    s2 = self.oh12(s2)
    s2 = self.oh12d(s2)
    s2 = self.oh2(s2)
    s2 = self.oh2d(s2)
    s2 = self.oh22(s2)
    s2 = self.oh22d(s2)
    s2 = self.oh3(s2)
    s2 = self.oh3d(s2)
    s2 = self.oh4(s2)
    s2 = self.oh4d(s2)
    return s2

  def getObjCenterInfo(self, batch_size, source):
    obj_pos = tf.reshape(source, (batch_size, self.target_maxlen, 12,3))
    obj_center = tf.reduce_mean(obj_pos, axis=-2)
    return obj_center[:, :, :]

  def call(self, inputs):
    print("in call")
    source = inputs[0]  #obj
    #target = inputs[1]  #h
    
    
    initialPos = self.getInitialPos(source)
    result = tf.expand_dims(initialPos, axis=1)

    for i in range(1, self.target_maxlen):
      lastObj = source[:, i-1, :]
      currentObj = source[:, i, :]
      lastHuman = result[:, -1, :]
      inputFeature = tf.concat([lastObj, currentObj, lastHuman], axis=-1)
      current_result = self.getFollowingPos(inputFeature)
      current_result = tf.expand_dims(current_result, axis=1)
      result = tf.concat([result, current_result], axis = 1)
    return result, initialPos
  
  def train_step(self, batch):
    source, target = batch
    target = tf.dtypes.cast(target, tf.float32)
    source = tf.dtypes.cast(source, tf.float32)
    y_initalPos = target[:, 0, :]
    obj_center = self.getObjCenterInfo(batch_size, source)
    y = tf.concat((target, obj_center), axis = -1)
    with tf.GradientTape() as tape:
      y_p, initalPos = self([source, target], training=True)
      loss = self.compiled_loss(y_true={"final_result": y, "intial_human":y_initalPos}, y_pred={"final_result":y_p, "intial_human":initalPos})
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    self.compiled_metrics.update_state(y_true={"final_result": y, "intial_human":y_initalPos}, y_pred={"final_result":y_p, "intial_human":initalPos})
    return {m.name: m.result() for m in self.metrics}

  def test_step(self, batch):
    source, target = batch
    target = tf.dtypes.cast(target, tf.float32)
    source = tf.dtypes.cast(source, tf.float32)
    y_initalPos = target[:, 0, :]
    y = target[:, :, :]

    preds, initalPos = self([source, target], training=False)
    self.compiled_metrics.update_state(y_true={"final_result": y, "intial_human":y_initalPos}, y_pred={"final_result":preds, "intial_human":initalPos})
    return {m.name: m.result() for m in self.metrics}
  
  def generate(self, testX):
    source = testX
    initialPos = self.getInitialPos(source)
    result = tf.expand_dims(initialPos, axis=1)

    for i in range(1, self.target_maxlen):
      lastObj = source[:, i-1, :]
      currentObj = source[:, i, :]
      lastHuman = result[:, -1, :]
      inputFeature = tf.concat([lastObj, currentObj, lastHuman], axis=-1)
      current_result = self.getFollowingPos(inputFeature)
      current_result = tf.expand_dims(current_result, axis=1)
      result = tf.concat([result, current_result], axis = 1)
    return result

  def generate_singleFrame(self, input_obj, input_human, useInitialPos=False):
    #source = testX
    if useInitialPos:
      assert len(input_obj.shape) == 3 #b,T,F
      initialPos = self.getInitialPos(input)
      return initialPos
    assert len(input_obj.shape) == 3 #b,T,F
    assert len(input_human.shape) == 2  #b, F
    
    lastObj = input_obj[:, 0, :]
    currentObj = input_obj[:, 1, :]
    inputFeature = tf.concat([lastObj, currentObj, input_human], axis=-1)
    current_result = self.getFollowingPos(inputFeature)
    return current_result
    
    