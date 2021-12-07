import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

batch_size = 4


class BaseEncoderDecoderLayout(keras.Model):
    def __init__(self, num_hid=128,dim_output=10):
        super().__init__()

        self.startFlatten = layers.Flatten()
        
        self.fc1 = layers.Dense(num_hid*4, activation='relu')
        self.d1 = layers.Dropout(0.1)
        self.b1 = layers.BatchNormalization()
        
        self.fc2 = layers.Dense(num_hid*3, activation='relu')
        self.d2 = layers.Dropout(0.1)
        self.b2 = layers.BatchNormalization()
        
        self.fc3 = layers.Dense(num_hid*2, activation='relu')
        self.d3 = layers.Dropout(0.1)
        self.b3 = layers.BatchNormalization()
        
        self.fc4 = layers.Dense(num_hid*2, activation='relu')
        self.d4 = layers.Dropout(0.1)
        self.b4 = layers.BatchNormalization()
        
        self.fc5 = layers.Dense(dim_output)

    def call(self, x):
        x = self.startFlatten(x)
        
        x = self.fc1(x)
        x = self.d1(x)
        x = self.b1(x)
        
        x = self.fc2(x)
        x = self.d2(x)
        x = self.b2(x)
        
        x = self.fc3(x)
        x = self.d3(x)
        x = self.b3(x)
        
        x = self.fc4(x)
        x = self.d4(x)
        x = self.b4(x)
        
        x = self.fc5(x)
        return x


class HOVAE(keras.Model):
    def __init__(self, target_maxlen = 85, human_InputDim = 84, human_OutputDim = 84, Obj_Dim = 36):
        super().__init__()
        
        self.target_maxlen = target_maxlen
        self.dim_z = 48
        self.INIT_POSE = BaseEncoderDecoderLayout(dim_output = self.human_OutputDim)
        self.ENCODER = BaseEncoderDecoderLayout(dim_output = self.dim_z * 2)
        self.PRIOR =  BaseEncoderDecoderLayout(dim_output = self.dim_z * 2)
        self.DECODER = BaseEncoderDecoderLayout(dim_output = self.human_OutputDim)
    
    def z_sample_kl(self, z, random=True):
        z_mean = z[:self.dim_z]
        z_log_var = z[self.dim_z:]
        kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var) + 1.0)
        if random:
            epsilon = tf.random.normal(shape=(batch_size, self.dim_z))
        else:
            epsilon = tf.ones(shape=(batch_size, self.dim_z)) * 0.0
        return z_mean + tf.math.exp(0.5 * z_log_var) * epsilon, kl_loss
    
    def getObjCenterInfo(self, batch_size, source):
        obj_pos = tf.reshape(source, (batch_size, self.target_maxlen, 12,3))
        obj_center = tf.reduce_mean(obj_pos, axis=-2)
        return obj_center[:, :, :]
    
    def call(self, inputs):
        objInput = inputs[0]
        humanInput = inputs[1]
        
        initialPos = self.INIT_POSE(objInput)
        result = tf.expand_dims(initialPos, axis=1)
        z_loss = 0.0
        
        for i in range(1, self.target_maxlen):
            lastObj = objInput[:, i-1, :]
            currentObj = objInput[:, i, :]
            #lastHuman = result[:, -1, :]
            lastHuman = humanInput[:, i-1, :]
            currentHuman = humanInput[:, i, :]
            
            lastFrameFeature = tf.concat([lastObj, lastHuman], axes=-1)
            currentFrameFeature = tf.concat([currentObj, currentHuman], axes=-1)
            
            z_encoder = self.ENCODER(tf.concat([currentFrameFeature, lastFrameFeature], axes=-1))
            z_prior = self.PRIOR(lastFrameFeature)
            z_loss += tf.keras.losses.MSE(z_encoder, z_prior)
            
            sample, kl_loss = self.z_sample_kl(z_encoder)
            
            current_result = self.DECODER(tf.concat([sample, lastFrameFeature], axis=-1))
            current_result = tf.expand_dims(current_result, axis=1)
            result = tf.concat([result, current_result], axis = 1)
        return result, initialPos, z_loss, kl_loss
    
    def train_step(self, batch):
        objInput, humanInput = batch
        objInput = tf.dtypes.cast(objInput, tf.float32)
        humanInput = tf.dtypes.cast(humanInput, tf.float32)
        y_initalPos = humanInput[:, 0, :]
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
    
    