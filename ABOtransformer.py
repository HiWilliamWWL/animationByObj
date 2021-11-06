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

    pe = tf.zeros((maxlen, self.num_hid))
    position = tf.range(start=0, limit=maxlen)
    position = tf.cast(position, tf.float32)
    div_term = tf.exp(tf.cast(tf.range(0, self.num_hid), tf.float32) * -(tf.math.log(10000.0) / self.num_hid))
    pe1 = tf.math.sin(tf.reshape(position,( maxlen, 1)) * tf.reshape( div_term, (1, self.num_hid)))
    pe2 = tf.math.cos(tf.reshape(position,( maxlen, 1)) * tf.reshape( div_term, (1, self.num_hid)))
    pattern1 = tf.convert_to_tensor(np.array([[1.0, 0.0] for i in range(self.num_hid//2)]).flatten(), dtype=tf.float32)
    pattern2 = tf.convert_to_tensor(np.array([[0.0, 1.0] for i in range(self.num_hid//2)]).flatten(), dtype=tf.float32)
    pe = pattern1 * pe1 + pattern2*pe2
    pe = tf.expand_dims(pe, 0)
    pe = tf.tile(pe,(batch_size, 1, 1))
    #positions = tf.range(start=0, limit=maxlen, delta=1)
    #positions = self.pos_emb(positions)
    result = x + pe
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
    #positions = tf.range(start=0, limit=maxlen, delta=1)
    #positions = self.pos_emb(positions)
    #result = x + positions
    #result = x

    pe = tf.zeros((maxlen, self.num_hid))
    position = tf.range(start=0, limit=maxlen)
    position = tf.cast(position, tf.float32)
    div_term = tf.exp(tf.cast(tf.range(0, self.num_hid), tf.float32) * -(tf.math.log(10000.0) / self.num_hid))
    pe1 = tf.math.sin(tf.reshape(position,( maxlen, 1)) * tf.reshape( div_term, (1, self.num_hid)))
    pe2 = tf.math.cos(tf.reshape(position,( maxlen, 1)) * tf.reshape( div_term, (1, self.num_hid)))
    pattern1 = tf.convert_to_tensor(np.array([[1.0, 0.0] for i in range(self.num_hid//2)]).flatten(), dtype=tf.float32)
    pattern2 = tf.convert_to_tensor(np.array([[0.0, 1.0] for i in range(self.num_hid//2)]).flatten(), dtype=tf.float32)
    pe = pattern1 * pe1 + pattern2*pe2
    pe = tf.expand_dims(pe, 0)
    pe = tf.tile(pe,(batch_size, 1, 1))

    result = x + pe
    return result

class TransformerEncoder(layers.Layer):
  def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
    super().__init__()
    self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    self.ffn = keras.Sequential(
        [
            layers.Dense(feed_forward_dim, activation="relu"),
            layers.Dense(embed_dim),
        ]
    )
    self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = layers.Dropout(rate)
    self.dropout2 = layers.Dropout(rate)

  def call(self, inputs, training):
    attn_output = self.att(inputs, inputs)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(inputs + attn_output)
    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=training)
    return self.layernorm2(out1 + ffn_output)

class TransformerDecoder(layers.Layer):
  def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
    super().__init__()
    self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
    self.self_att = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim
    )
    self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    self.self_dropout = layers.Dropout(0.1)
    self.enc_dropout = layers.Dropout(0.1)
    self.ffn_dropout = layers.Dropout(0.1)
    self.ffn = keras.Sequential(
        [
            layers.Dense(feed_forward_dim, activation="relu"),
            layers.Dense(embed_dim),
        ]
    )

  def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
    """Masks the upper half of the dot product matrix in self attention.

    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    mask = tf.tile(mask, mult)
    return mask
  
  def maskGenerate2(self, target, max_length):
    i = tf.range(max_length)[:, None]
    j = tf.range(max_length)
    m = i >= j
    mask = tf.cast(m, tf.bool)
    mask = tf.reshape(mask, [1, max_length, max_length])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    final_mask = tf.Variable(tf.tile(mask, mult))
    
    zerosPattern = tf.zeros_like(target[:,:,0])
    #zerosPattern = tf.zeros((batch_size, 129))
    mask = tf.reduce_sum(tf.math.abs(target), axis=-1) > zerosPattern
    mask2 = mask[:,:]
    final_mask = None
    for i in range(batch_size):
      lastFrame = tf.cast( tf.reduce_sum(tf.cast(mask2[i], tf.float32)), tf.int32)
      final_mask[i, lastFrame:, :] = False
      final_mask[i, :, lastFrame:] = False
    print(final_mask)
    return final_mask

  def call(self, enc_out, target):
    input_shape = tf.shape(target)
    batch_size = input_shape[0]
    seq_len = input_shape[1]
    #tf.print(seq_len)
    causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
    #tf.print(causal_mask.shape)
    #print(target.shape)
    #exit()
    #causal_mask = self.maskGenerate2(target, seq_len)
    
    target_att = self.self_att(target, target, attention_mask=causal_mask)
    target_norm = self.layernorm1(target + self.self_dropout(target_att))
    enc_out = self.enc_att(target_norm, enc_out)
    enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
    ffn_out = self.ffn(enc_out_norm)
    ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
    return ffn_out_norm


class Transformer(keras.Model):
  def __init__(
      self, num_hid=64, num_head=2, num_feed_forward=128, source_maxlen=60, target_maxlen=60, num_layers_enc=4, num_layers_dec=1, num_classes=10):
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
              TransformerEncoder(num_hid, num_head, num_feed_forward)
              for _ in range(num_layers_enc)
          ]
      )
      for i in range(num_layers_dec):
        setattr(
              self,
              f"dec_layer_{i}",
              TransformerDecoder(num_hid, num_head, num_feed_forward))
      self.classifier1 = layers.Dense(num_hid)
      self.classifier2 = layers.Dense(num_classes, name="final_result")
      print("finish process5")

  def decode(self, enc_out, target):
    y = self.dec_input(target)
    for i in range(self.num_layers_dec):
      y = getattr(self, f"dec_layer_{i}")(enc_out, y)
    return y

  def call(self, inputs):
    print("in call")
    source = inputs[0]
    target = inputs[1]
    
    
    x = self.encoder(source)
    y = self.decode(x, target)
    y = self.classifier1(y)
    result = self.classifier2(y)
    return result
  
    

  def train_step(self, batch):
    """Processes one batch inside model.fit()."""
    print("using oritnal train step")
    source, target = batch
    dec_input = target[:, :-1, :]
    dec_target = target[:, 1:, :]
    y = dec_target
    with tf.GradientTape() as tape:
        y_pred = self([source, dec_input], training=True)
        #loss = self.compiled_loss(dec_target, y_pred=preds, regularization_losses=self.losses)
        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #self.compiled_metrics.update_state(loss)
    self.compiled_metrics.update_state(y, y_pred)
    #return {"loss": self.loss_metric.result()}
    return {m.name: m.result() for m in self.metrics}

  '''
  def train_step_notUsing(self, batch):
    """Processes one batch inside model.fit()."""
    source, target = batch
    dec_input = target[:, :-1]
    dec_target = target[:, 1:]
    y = dec_target
    dec2_input = dec_input[:, 0, :]
    dec2_input = tf.expand_dims(dec2_input, axis=-2)
    dec2_input = tf.dtypes.cast(dec2_input, tf.float32)
    with tf.GradientTape() as tape:
      enc = self.encoder(source)
      
      for i in range(self.target_maxlen - 1):
        dec_out = self.decode(enc, dec2_input)
        
        out = self.classifier(dec_out)
        
        out = out[:, -1, :]
        out = tf.expand_dims(out, axis=1)
        dec2_input = tf.concat([dec2_input, out], axis=-2)
        
      loss = self.compiled_loss(y, dec2_input[:, -1, :])
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #self.compiled_metrics.update_state(loss)
    self.compiled_metrics.update_state(y, dec2_input[:, -1, :])
    #return {"loss": self.loss_metric.result()}
    return {m.name: m.result() for m in self.metrics}
  '''
  
  def test_step(self, batch):
    print("using oritnal test step")
    x, y = batch
    source = x
    target = y
    dec_input = target[:, :-1]
    dec_target = target[:, 1:]

    preds = self([source, dec_input])
    loss = self.compiled_loss(dec_target, y_pred=preds)
    return {m.name: m.result() for m in self.metrics}

  def test_step2(self, batch):
    x, y = batch
    source = x
    target = y
    dec_input = target[:, :-1]
    dec_target = target[:, 1:]

    preds = self([source, dec_input])
    loss = self.compiled_loss(dec_target, y_pred=preds)
    print(loss)
    return preds

  def generate2(self, source, dec_start_input, dec_Pre= None):
    print("using oritnal generate step")
    source = source.reshape((1, self.target_maxlen, 36))
    enc = self.encoder(source)
    print("enter the loop")
    for i in range(self.target_maxlen - 1):
      dec_out = self.decode(enc, dec_start_input)
      #print("decode sccuess")
      #print(dec_out.shape)
      out = self.classifier1(dec_out)
      out = self.classifier2(out)
      if i == self.target_maxlen - 2:
        result = tf.identity(out)
      out = out[:, -1, :]
      out = tf.expand_dims(out, axis=1)
      #print(out.shape)
      #print("end")
      dec_start_input = tf.concat([dec_start_input, out], axis=-2)
      #print(dec_start.shape)
      #print("end loop")
      #print()
    #loss = self.compiled_loss(dec_start_input[:,1:,:], y_pred=dec_Pre[:,:,:])
    #print(loss)
    print("++++++++")
    #return result
    return dec_start_input[:, 1:, :]
  
  def generate(self, source, dec_start_input, dec_Pre= None):
    print("using oritnal generate step")
    source = source.reshape((1, self.target_maxlen, 36))
    enc = self.encoder(source)
    print("enter the loop")
    for i in range(self.target_maxlen - 1):
      dec_out = self.decode(enc, dec_start_input)
      out = self.classifier1(dec_out)
      out = self.classifier2(out)
      out = out[:, -1, :]
      out = tf.expand_dims(out, axis=1)
      dec_start_input = tf.concat([dec_start_input, out], axis=-2)
    dec_start_input = tf.dtypes.cast(dec_start_input, tf.float32)
    #dec2_input2 = tf.concat((tf.convert_to_tensor(dec_start_input, dtype=tf.float32), result[:,:-1,:]), axis=1)
    result = self([source, dec_start_input[:, :-1, :]])
    print("++++++++")
    return result

class Transformer_pre2pre(Transformer):
  def __init__(
        self, num_hid=64, num_head=2, num_feed_forward=128, source_maxlen=60, target_maxlen=60, num_layers_enc=4, num_layers_dec=1, num_classes=10):
        super().__init__(num_hid, num_head, num_feed_forward, source_maxlen, target_maxlen, num_layers_enc, num_layers_dec, num_classes)
  
  def getObjCenterInfo(self, batch_size, source):
    obj_pos = tf.reshape(source, (batch_size, self.target_maxlen, 12,3))
    obj_final_pos = obj_pos[:, 0, :,:]
    obj_final_center = tf.reduce_mean(obj_final_pos, axis=-2)
    obj_final_center = tf.expand_dims(obj_final_center, axis=1)
    obj_final_center = tf.tile(obj_final_center, (1,self.target_maxlen-1,1))
    return obj_final_center

  def train_step2(self, batch):
    """Processes one batch inside model.fit()."""
    print("using new train step, once")
    source, target = batch
    dec_input = target[:, :-1]
    dec_target = target[:, 1:]
    y = dec_target
    obj_center = self.getObjCenterInfo(batch_size, source)
    y = tf.concat((y, obj_center), axis = -1)

    dec2_input = dec_input[:, 0, :]
    dec2_input = tf.expand_dims(dec2_input, axis=-2)
    dec2_input = tf.dtypes.cast(dec2_input, tf.float32)
    #with tf.GradientTape() as tape:
    enc = self.encoder(source)
    for i in range(self.target_maxlen - 2):
      dec_out = self.decode(enc, dec2_input)
      
      out = self.classifier1(dec_out)
      out = self.classifier2(out)
      out = out[:, -1, :]
      out = tf.expand_dims(out, axis=1)
      dec2_input = tf.concat([dec2_input, out], axis=-2)
    with tf.GradientTape() as tape:
      enc = self.encoder(source)
      dec_out = self.decode(enc, dec2_input)
      out = self.classifier1(dec_out)
      out = self.classifier2(out)
      out = out[:, -1, :]
      out = tf.expand_dims(out, axis=1)
      dec2_input = tf.concat([dec2_input, out], axis=-2)
      result = dec2_input[:, 1:, :]
      loss = self.compiled_loss(y, result)
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #self.compiled_metrics.update_state(loss)
    self.compiled_metrics.update_state(y, result)
    #return {"loss": self.loss_metric.result()}
    return {m.name: m.result() for m in self.metrics}

  def train_step(self, batch):
    """Processes one batch inside model.fit()."""
    print("using new train step, twice")
    source, target = batch
    dec_input = target[:, :-1]
    dec_target = target[:, 1:]
    y = dec_target
    obj_center = self.getObjCenterInfo(batch_size, source)
    y = tf.concat((y, obj_center), axis = -1)

    dec2_input = dec_input[:, 0, :]
    dec2_input = tf.expand_dims(dec2_input, axis=-2)
    dec2_input = tf.dtypes.cast(dec2_input, tf.float32)
    #with tf.GradientTape() as tape:
    enc = self.encoder(source)
    for i in range(self.target_maxlen - 1):
      dec_out = self.decode(enc, dec2_input)
      
      out = self.classifier1(dec_out)
      out = self.classifier2(out)
      out = out[:, -1, :]
      out = tf.expand_dims(out, axis=1)
      dec2_input = tf.concat([dec2_input, out], axis=-2)
    #i = self.target_maxlen - 2
    #dec_input = tf.dtypes.cast(dec2_input, tf.float32)
    #dec2_input2 = tf.concat((tf.expand_dims(dec_input[:,0,:], axis=1), result[:,:-1,:]), axis=1)
    dec2_input2 = dec2_input[:, :-1, :]
    with tf.GradientTape() as tape:
      result = self([source, dec2_input2], training=True)
      print("++++++++++++++")
      print(result.shape)
      print(y.shape)
      loss = self.compiled_loss(y, result)
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #self.compiled_metrics.update_state(loss)
    self.compiled_metrics.update_state(y, result)
    #return {"loss": self.loss_metric.result()}
    return {m.name: m.result() for m in self.metrics}




class Transformer_newScheduleSampling(Transformer):
  def __init__(
        self, num_hid=64, num_head=2, num_feed_forward=128, source_maxlen=60, target_maxlen=60, num_layers_enc=4, num_layers_dec=1, num_classes=10, scheule_sampling_gt_rate = 80):
        super().__init__(num_hid, num_head, num_feed_forward, source_maxlen, target_maxlen, num_layers_enc, num_layers_dec, num_classes)
        self.scheule_sampling_gt_rate = scheule_sampling_gt_rate
  
  def getObjCenterInfo_finalFrame(self, batch_size, source):
    obj_pos = tf.reshape(source, (batch_size, self.target_maxlen, 12,3))
    obj_final_pos = obj_pos[:, 0, :,:]
    obj_final_center = tf.reduce_mean(obj_final_pos, axis=-2)
    obj_final_center = tf.expand_dims(obj_final_center, axis=1)
    obj_final_center = tf.tile(obj_final_center, (1,self.target_maxlen-1,1))
    return obj_final_center
  
  def getObjCenterInfo(self, batch_size, source):
    obj_pos = tf.reshape(source, (batch_size, self.target_maxlen, 12,3))
    obj_center = tf.reduce_mean(obj_pos, axis=-2)
    #obj_final_center = tf.expand_dims(obj_final_center, axis=1)
    #obj_final_center = tf.tile(obj_final_center, (1,self.target_maxlen-1,1))
    return obj_center[:, 1:, :]

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



  def train_step(self, batch):
    """Gt2Pre works well with mean pose loss"""
    print("using new train step 2")
    source, target = batch
    dec_input = target[:, :-1, :]
    dec_input = tf.dtypes.cast(dec_input, tf.float32)
    dec_target = target[:, 1:, :]
    y = dec_target
    obj_center = self.getObjCenterInfo(batch_size, source)
    y = tf.concat((y, obj_center), axis = -1)
    #with tf.GradientTape() as tape:
    y_pred1 = self([source, dec_input], training=True)
    #loss1 = self.compiled_loss(y, y_pred1)
    new_input = tf.concat((tf.expand_dims(dec_input[:,0,:], axis=1), y_pred1[:,:-1,:]), axis=1)
    new_input = self.mix_Data(dec_input, new_input, self.scheule_sampling_gt_rate)
    with tf.GradientTape() as tape:
      y_pred2 = self([source, new_input], training=True)
      loss2 = self.compiled_loss(y, y_pred2)

      """
      -- Jun 21, 23:34
      I think you need to read again the paper Sceduled Samping for Transformers (https://arxiv.org/abs/1906.07651)
      In that paper and in the earlier paper by Bengio et al., 2015, 
      the GT history and the prediction of the first decoder are mixed according to some kind of scheduled plan
      
      Averaging loss1 and loss2 is too simple, I guess. 
      So, you need to re-implement that algorithm described in paper Sceduled Samping for Transformers
      to fully understand if the problem is actually caused by the model exposured to the GT history.
      """
      #loss = loss1 + loss2
      loss = loss2
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #self.compiled_metrics.update_state(loss)
    self.compiled_metrics.update_state(y, y_pred2)
    #return {"loss": self.loss_metric.result()}
    return {m.name: m.result() for m in self.metrics}
    
  def test_step2(self, batch):
    global batch_size
    x, y = batch
    source = x
    target = y
    dec_input = target[:, :-1]
    dec_target = target[:, 1:]

    preds = self([source, dec_input])

    metric = losses.maskMSE(dec_target, preds)
    print("-------metrics-------------")
    print(metric)
    return preds


class Transformer_initalPose(Transformer_newScheduleSampling):
  def __init__(
        self, num_hid=64, num_head=2, num_feed_forward=128, source_maxlen=60, target_maxlen=60, num_layers_enc=4, num_layers_dec=1, num_classes=10, scheule_sampling_gt_rate = 80):
        super().__init__(num_hid, num_head, num_feed_forward, source_maxlen, target_maxlen, num_layers_enc, num_layers_dec, num_classes, scheule_sampling_gt_rate)
        self.initalPosStart1 = layers.Dense(num_hid*2, activation='relu')
        self.initalPosStart2 = layers.Dense(num_hid*2, activation='relu')
        self.initalPosStart25 = layers.Dense(num_hid)
        self.initalPosStart3 = layers.Dense(63, name="intial_human")
  

  def call(self, inputs):
    print("in call")
    source = inputs[0]
    target = inputs[1]

    s1 = self.initalPosStart1(source[:,0,:])
    s2 = self.initalPosStart2(s1)
    s2 = self.initalPosStart25(s2)
    initalPos = self.initalPosStart3(s2)
    target = tf.concat((tf.expand_dims(initalPos, axis=1), target[:, 1:, :]), axis=1)
    
    x = self.encoder(source)
    y = self.decode(x, target)
    y = self.classifier1(y)
    result = self.classifier2(y)
    return result, initalPos
  
  def train_step(self, batch):
    """Gt2Pre works well with mean pose loss"""
    print("using new train step 2 Transformer_initalPose")
    source, target = batch
    dec_input = target[:, :-1, :]
    dec_input = tf.dtypes.cast(dec_input, tf.float32)
    dec_target = target[:, 1:, :]

    y = dec_target
    y_initalPos = target[:, 0, :]

    obj_center = self.getObjCenterInfo(batch_size, source)
    y = tf.concat((y, obj_center), axis = -1)
    #with tf.GradientTape() as tape:
    y_pred1, initalPos = self([source, dec_input], training=True)
    #loss1 = self.compiled_loss(y, y_pred1)
    new_input = tf.concat((tf.expand_dims(dec_input[:,0,:], axis=1), y_pred1[:,:-1,:]), axis=1)
    new_input = self.mix_Data(dec_input, new_input, self.scheule_sampling_gt_rate)
    with tf.GradientTape() as tape:
      y_pred2, initalPos = self([source, new_input], training=True)
      
      loss2 = self.compiled_loss(y_true={"final_result": y, "intial_human":y_initalPos}, y_pred={"final_result":y_pred2, "intial_human":initalPos})
      #print(loss2)
      #exit()
      #loss = loss1 + loss2
      loss = loss2
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #self.compiled_metrics.update_state(loss)
    self.compiled_metrics.update_state(y_true={"final_result": y, "intial_human":y_initalPos}, y_pred={"final_result":y_pred2, "intial_human":initalPos})
    #return {"loss": self.loss_metric.result()}
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
    print("using oritnal generate step")
    source = source.reshape((1, self.target_maxlen, 36))
    enc = self.encoder(source)
    s1 = self.initalPosStart1(source[:,0,:])
    s2 = self.initalPosStart2(s1)
    s2 = self.initalPosStart25(s2)
    initalPos = self.initalPosStart3(s2)
    dec_start_input = tf.expand_dims(initalPos, axis=1)
    #target = tf.concat((tf.expand_dims(initalPos, axis=1), target[:, 1:, :]), axis=1)
    for i in range(self.target_maxlen - 1):
      dec_out = self.decode(enc, dec_start_input)
      out = self.classifier1(dec_out)
      out = self.classifier2(out)
      out = out[:, -1, :]
      out = tf.expand_dims(out, axis=1)
      dec_start_input = tf.concat([dec_start_input, out], axis=-2)
    dec_start_input = tf.dtypes.cast(dec_start_input, tf.float32)
    #dec2_input2 = tf.concat((tf.convert_to_tensor(dec_start_input, dtype=tf.float32), result[:,:-1,:]), axis=1)
    result = self([source, dec_start_input[:, :-1, :]])
    print("++++++++")
    return result

class Transformer_VAEinitalPose(Transformer_initalPose):
  def __init__(
        self, num_hid=64, num_head=2, num_feed_forward=128, source_maxlen=60, target_maxlen=60, num_layers_enc=4, num_layers_dec=1, num_classes=10, scheule_sampling_gt_rate = 80):
        super().__init__(num_hid, num_head, num_feed_forward, source_maxlen, target_maxlen, num_layers_enc, num_layers_dec, num_classes, scheule_sampling_gt_rate)
        self.latentDimension = num_hid
        self.add2Start = tf.Variable(tf.random.normal(shape=(1, 2, 36)), trainable=True)
        self.discriminator_model = None
        #self.add2Start_batch = tf.tile(self.add2Start, [batch_size, 1, 1])

  def sampling(self, z_mean, z_log_var, random=True):
    #batch_size self.latentDimension
    if random:
      epsilon = tf.random.normal(shape=(batch_size, self.latentDimension))
    else:
      epsilon = tf.ones(shape=(batch_size, self.latentDimension)) * 0.0
    return z_mean + tf.math.exp(0.5 * z_log_var) * epsilon
  

  def setDiscriminator(self, model_using):
    self.discriminator_model = model_using


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
    #source_add2 = tf.concat([add2Start_batch, source], axis=1)   #change the order???
    source_add2 = tf.concat([source, add2Start_batch], axis=1)
    
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
      if self.discriminator_model is not None:
        loss_discriminator = tf.reduce_sum(self.discriminator_model(y_pred2))
        tf.print("dis")
        tf.print(self.discriminator_model(y_pred2))
        tf.print(loss_discriminator)
        loss += loss_discriminator
      #tf.print(firstTwoFrameDiff)
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    self.compiled_metrics.update_state(y_true={"final_result": y, "intial_human":y_initalPos}, y_pred={"final_result":y_pred2, "intial_human":initalPos})
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

    #source_add2 = tf.concat([self.add2Start, source], axis=1)
    source_add2 = tf.concat([source, self.add2Start], axis=1)
    x = self.encoder(source_add2)
    

    z_mean = x[:, 0, :]
    z_log_var = x[:, 1, :]
    sample = self.sampling(z_mean, z_log_var, False)
    sample = tf.tile(tf.expand_dims(sample, axis=1), [1, self.target_maxlen, 1])


    #enc = tf.concat((x[:, 2:, :], sample), axis=-1)  #VAE2 test
    enc = tf.concat((source, sample), axis=-1)

    for i in range(self.target_maxlen - 1):
      dec_out = self.decode(enc, dec_start_input)
      out = self.classifier1(dec_out)
      out = self.classifier2(out)
      out = out[:, -1, :]

      '''    #use IK as post-process
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
      '''
      
      out = tf.expand_dims(out, axis=1)
      dec_start_input = tf.concat([dec_start_input, out], axis=-2)
    dec_start_input = tf.dtypes.cast(dec_start_input, tf.float32)
    #dec2_input2 = tf.concat((tf.convert_to_tensor(dec_start_input, dtype=tf.float32), result[:,:-1,:]), axis=1)
    #result = self([source, dec_start_input[:, :-1, :]])[0]
    result = dec_start_input[:, :-1, :]
    print("++++++++")
    return result

  def predict_step(self, batch):
    source, target = batch
    source = tf.expand_dims(source, axis=0)
    target = tf.expand_dims(target, axis=0)
    dec_input = target[:, :-1, :]
    dec_input = tf.dtypes.cast(dec_input, tf.float32)
    dec_target = target[:, 1:, :]

    y = dec_target
    y_initalPos = target[:, 0, :]

    obj_center = self.getObjCenterInfo(batch_size, source)
    #y = tf.concat((y, obj_center), axis = -1)
    #with tf.GradientTape() as tape:
    y_pred1, _, _ = self([source, dec_input], training=False)
    #loss1 = self.compiled_loss(y, y_pred1)
    new_input = tf.concat((tf.expand_dims(dec_input[:,0,:], axis=1), y_pred1[:,:-1,:]), axis=1)
    new_input = self.mix_Data(dec_input, new_input, self.scheule_sampling_gt_rate)
    y_pred2, _, _ = self([source, new_input], training=False)
    return y_pred2

