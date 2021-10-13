import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import dataLoader
loader = dataLoader.trainDataLoader()
loader.prepareDataset()
batch_size = 4

import losses

import thisIK

class TokenEmbeddingObj(layers.Layer):
  def __init__(self, maxlen, num_hid):
    super().__init__()
    self.denseEmb1 = tf.keras.layers.Dense(64)
    self.denseEmb2 = tf.keras.layers.Dense(64, activation='relu')
    self.denseEmb3 = tf.keras.layers.Dense(num_hid)
    self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)
    self.num_hid = num_hid

  def call(self, x):
    maxlen = x.shape[-2]
    x = self.denseEmb1(x)
    x = self.denseEmb2(x)
    x = self.denseEmb3(x)
    result = x
    return result

class TokenEmbeddingPpl(layers.Layer):
  def __init__(self, maxlen, num_hid):
    super().__init__()
    self.denseEmb1 = tf.keras.layers.Dense(128)
    self.denseEmb2 = tf.keras.layers.Dense(num_hid, activation='relu')
    self.denseEmb3 = tf.keras.layers.Dense(num_hid)
    self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)
    self.num_hid = num_hid

  def call(self, x):
    maxlen = x.shape[-2]
    x = self.denseEmb1(x)
    x = self.denseEmb2(x)
    x = self.denseEmb3(x)

    result = x
    return result

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = keras.backend.dot(inputs, self.kernel)
        output = h + keras.backend.dot(prev_output, self.recurrent_kernel)
        return output, [output]


class baseFC_RNN(keras.Model):
  def __init__(
        self, num_hid=64, num_head=2, num_feed_forward=128, source_maxlen=60, target_maxlen=60, num_layers_enc=4, num_layers_dec=1, num_classes=10, scheule_sampling_gt_rate = 80):
        super().__init__()
        #print(self.metrics)
        #self.loss_metric = keras.metrics.Mean(name="loss")
        #self.loss_metric = tf.keras.metrics.MeanSquaredError(name="loss")
        #self.metrics = [self.loss_metric]
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes
        #self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)  ###
        self.enc_input = TokenEmbeddingObj(num_hid=num_hid, maxlen=target_maxlen)
        self.dec_input = TokenEmbeddingPpl(maxlen=target_maxlen, num_hid=num_hid)
        self.encoder = keras.Sequential([self.enc_input]
          + [
              layers.Dense(num_hid)
              for _ in range(num_layers_enc)
          ]
        )

        self.cells = [MinimalRNNCell(num_hid) for _ in range(num_layers_dec)]
        setattr(
              self,
              f"dec_layer_0",
              layers.RNN(self.cells, return_sequences = True))

        self.classifier1 = layers.Dense(num_hid)
        self.classifier2 = layers.Dense(num_classes, name="final_result")
        print("finish process5")
        
        #super().__init__(num_hid, num_head, num_feed_forward, source_maxlen, target_maxlen, num_layers_enc, num_layers_dec, num_classes, scheule_sampling_gt_rate)
        self.latentDimension = num_hid
        self.add2Start = tf.Variable(tf.random.normal(shape=(1, 2, 36)), trainable=True)
        self.initalPosStart1 = layers.Dense(num_hid*2, activation='relu')
        self.initalPosStart2 = layers.Dense(num_hid*2, activation='relu')
        self.initalPosStart25 = layers.Dense(num_hid)
        self.initalPosStart3 = layers.Dense(63, name="intial_human")

        self.scheule_sampling_gt_rate = scheule_sampling_gt_rate
  
  def decode(self, enc_out, target):
    y = self.dec_input(target)
    #print(enc_out)
    #print(y)
    finalInput = tf.concat([enc_out[:, 1:, :], y], axis=-1)
    y = getattr(self, f"dec_layer_0")(finalInput)
    return y

  def sampling(self, z_mean, z_log_var, random=True):
    #batch_size self.latentDimension
    if random:
      epsilon = tf.random.normal(shape=(batch_size, self.latentDimension))
    else:
      epsilon = tf.ones(shape=(batch_size, self.latentDimension)) * 0.0
    return z_mean + tf.math.exp(0.5 * z_log_var) * epsilon
  
  def getObjCenterInfo(self, batch_size, source):
    obj_pos = tf.reshape(source, (batch_size, self.target_maxlen, 12,3))
    obj_final_pos = obj_pos[:, 0, :,:]
    obj_final_center = tf.reduce_mean(obj_final_pos, axis=-2)
    obj_final_center = tf.expand_dims(obj_final_center, axis=1)
    obj_final_center = tf.tile(obj_final_center, (1,self.target_maxlen-1,1))
    return obj_final_center
  
  def mix_Data(self, dec_input, pre_input, gt_percent):
    resultShape = [batch_size * (self.target_maxlen-1) * self.num_classes // 3]
    randomChoice = tf.random.uniform(shape=resultShape, maxval=100.0)
    randomChoice = randomChoice < gt_percent
    randomChoice = tf.reshape(randomChoice, (batch_size, (self.target_maxlen-1), self.num_classes // 3, 1))
    randomChoice = tf.tile(randomChoice, [1,1,1,3])
    randomChoice = tf.reshape(randomChoice, (batch_size, (self.target_maxlen-1), self.num_classes))
    not_randomChoice = tf.math.logical_not(randomChoice)
    randomChoice = tf.cast(randomChoice, tf.dtypes.float32)
    not_randomChoice = tf.cast(not_randomChoice, tf.dtypes.float32)
    #tf.print(not_randomChoice)
    result = tf.zeros_like(dec_input)
    result = result + randomChoice * dec_input + not_randomChoice * pre_input
    return result


  def call(self, inputs):
    print("in call")
    source = inputs[0]
    target = inputs[1]

    s1 = self.initalPosStart1(source[:,0,:])
    s2 = self.initalPosStart2(s1)
    s2 = self.initalPosStart25(s2)
    initalPos = self.initalPosStart3(s2)
    target = tf.concat((tf.expand_dims(initalPos, axis=1), target[:, 1:, :]), axis=1)

    initalPos2 = tf.identity(initalPos)
    firstTwoFrameDiff = tf.keras.losses.mean_squared_error(initalPos2, target[:,1,:])

    add2Start_batch = tf.tile(self.add2Start, [batch_size, 1, 1])
    source_add2 = tf.concat([add2Start_batch, source], axis=1)
    
    x = self.encoder(source_add2)
    z_mean = x[:, 0, :]
    z_log_var = x[:, 1, :]
    kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var) + 1.0)
    #self.add_loss(kl_loss)
    #tf.print(kl_loss)

    sample = self.sampling(z_mean, z_log_var)
    sample = tf.tile(tf.expand_dims(sample, axis=1), [1, self.target_maxlen, 1])

    #x = x[:, 2:, :]

    #x = tf.concat((x[:, 2:, :], sample), axis=-1)  #VAE2 test
    x = tf.concat((source, sample), axis=-1)

    y = self.decode(x, target)
    y = self.classifier1(y)
    result = self.classifier2(y)
    return result, initalPos, (kl_loss, firstTwoFrameDiff) 

  def train_step(self, batch):
    """Gt2Pre works well with mean pose loss"""
    print("using new train step 2 Transformer_VAEinitalPose")
    source, target = batch
    dec_input = target[:, :-1, :]
    dec_input = tf.dtypes.cast(dec_input, tf.float32)
    dec_target = target[:, 1:, :]

    y = dec_target
    y_initalPos = target[:, 0, :]

    obj_center = self.getObjCenterInfo(batch_size, source)
    y = tf.concat((y, obj_center), axis = -1)
    #with tf.GradientTape() as tape:
    y_pred1, _, _ = self([source, dec_input], training=True)
    #loss1 = self.compiled_loss(y, y_pred1)
    new_input = tf.concat((tf.expand_dims(dec_input[:,0,:], axis=1), y_pred1[:,:-1,:]), axis=1)
    new_input = self.mix_Data(dec_input, new_input, self.scheule_sampling_gt_rate)
    with tf.GradientTape() as tape:
      y_pred2, initalPos, internal_losses = self([source, new_input], training=True)
      kl_loss, firstTwoFrameDiff = internal_losses
      loss2 = self.compiled_loss(y_true={"final_result": y, "intial_human":y_initalPos}, y_pred={"final_result":y_pred2, "intial_human":initalPos})
      loss = loss2 + kl_loss + firstTwoFrameDiff
      tf.print("two frame diff")
      #tf.print(firstTwoFrameDiff)
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    self.compiled_metrics.update_state(y_true={"final_result": y, "intial_human":y_initalPos}, y_pred={"final_result":y_pred2, "intial_human":initalPos})
    return {m.name: m.result() for m in self.metrics}


  def test_step(self, batch):
    print("using oritnal test step")
    x, y = batch
    source = x
    target = y
    y = target[:, 1:, :]
    dec_input = target[:, :-1]
    dec_target = target[:, 1:]
    y_initalPos = target[:, 0, :]

    preds, initalPos = self([source, dec_input])[:2]
    #loss = self.compiled_loss(dec_target, y_pred=preds)
    self.compiled_metrics.update_state(y_true={"final_result": y, "intial_human":y_initalPos}, y_pred={"final_result":preds, "intial_human":initalPos})
    return {m.name: m.result() for m in self.metrics}


  def generate(self, source, dec_start_input, dec_Pre= None):
    print("using VAE generate step")
    source = source.reshape((1, self.target_maxlen, 36))
    #enc = self.encoder(source)
    s1 = self.initalPosStart1(source[:,0,:])
    s2 = self.initalPosStart2(s1)
    s2 = self.initalPosStart25(s2)
    initalPos = self.initalPosStart3(s2)
    dec_start_input = tf.expand_dims(initalPos, axis=1)

    source_add2 = tf.concat([self.add2Start, source], axis=1)
    x = self.encoder(source_add2)
    z_mean = x[:, 0, :]
    z_log_var = x[:, 1, :]

    sample = self.sampling(z_mean, z_log_var, False)
    sample = tf.tile(tf.expand_dims(sample, axis=1), [1, self.target_maxlen, 1])

    #enc = tf.concat((x[:, 2:, :], sample), axis=-1)  #VAE2 test
    enc = tf.concat((source, sample), axis=-1)

    for i in range(self.target_maxlen - 1):
      dec_out = self.decode(enc[:, :i+2, :], dec_start_input)
      out = self.classifier1(dec_out)
      out = self.classifier2(out)
      out = out[:, -1, :]

      #'''  #cancel IK
      outCentered = out[:,:].numpy() * loader.ppl_std  + loader.ppl_mean
      #6-7-8  10-11-12
      shoulder1, shoulder2 = (outCentered[0, 6*3:6*3+3], outCentered[0, 10*3: 10*3+3])
      elbow1, elbow2 = (outCentered[0, 7*3: 7*3+3], outCentered[0, 11*3:11*3+3])
      hand1, hand2 = (outCentered[0, 8*3:8*3+3], outCentered[0, 12*3:12*3+3])
      newelbow1, newhand1 = thisIK.solver(source[0, i, 7*3:7*3+3] - outCentered[0, :3], shoulder1, elbow1, hand1)
      newelbow2, newhand2 = thisIK.solver(source[0, i, 9*3:9*3+3] - outCentered[0, :3], shoulder2, elbow2, hand2)
      outCentered[0,7*3: 7*3+3], outCentered[0,8*3: 8*3+3] = (newelbow1, newhand1)
      outCentered[0,11*3: 11*3+3], outCentered[0,12*3: 12*3+3] = (newelbow2, newhand2)
      out = tf.convert_to_tensor((outCentered - loader.ppl_mean) / loader.ppl_std, dtype=tf.float32)
      #'''
      
      out = tf.expand_dims(out, axis=1)
      dec_start_input = tf.concat([dec_start_input, out], axis=-2)
    dec_start_input = tf.dtypes.cast(dec_start_input, tf.float32)
    #dec2_input2 = tf.concat((tf.convert_to_tensor(dec_start_input, dtype=tf.float32), result[:,:-1,:]), axis=1)
    #result = self([source, dec_start_input[:, :-1, :]])[0]
    result = dec_start_input[:, :-1, :]
    print("++++++++")
    return result