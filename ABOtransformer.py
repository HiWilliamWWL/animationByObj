import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import dataLoader

loader = dataLoader.trainDataLoader()
loader.prepareDataset()
batch_size = 2

def maskLabelLoss(yTrue, yPred):
  #hand 8,12 base-0
  print("enter loss")
  obj_center = None
  in_train = False
  add_hand = True
  if yTrue.shape[2] > 63:
    obj_center = yTrue[:, -1, -3:]
    yTrue = yTrue[:, :, :-3]
    in_train = True
  zerosPattern = tf.zeros_like(yPred[:,:,0])
  #zerosPattern = tf.zeros((batch_size, 129))
  mask = tf.reduce_sum(tf.math.abs(yTrue), axis=-1) > zerosPattern
  mask2 = mask[:,:]
  mask = tf.expand_dims(mask, axis=-1)
  mask = tf.tile(mask, [1,1,63])
  print("&&&&&")
  print(yPred.shape)
  print(mask.shape)
  
  yTrue_m = tf.boolean_mask(yTrue, mask)
  yPred_m = tf.boolean_mask(yPred, mask)
  mes = tf.keras.losses.MSE(yTrue_m, yPred_m)
  
  #loader.pose_mean  loader.pose_cov_inv
  groupedPre = tf.reshape(tf.identity(yPred_m), (-1, 63))
  part = groupedPre[:, 3:] - tf.convert_to_tensor(loader.pose_mean, dtype=tf.float32)

  handLoss = 0.0
  if in_train and add_hand:
    for i in range(batch_size):
      lastFrame = tf.cast( tf.reduce_sum(tf.cast(mask2[i], tf.float32)), tf.int32)
      handPos1 = yPred[i, lastFrame-2, 8*3:8*3+3]
      handPos1 += yPred[i, lastFrame-2, :3]
      handPos1 /= 5.0 #norm
      diss1 = tf.math.sqrt(tf.reduce_sum((handPos1 - obj_center[i])**2))
      if diss1 > 0.5:
        handLoss += diss1 * 1.5
      handPos2 = yPred[i, lastFrame-2, 12*3:12*3+3]
      handPos2 += yPred[i, lastFrame-2, :3]
      handPos2 /= 5.0 #norm
      diss2 = tf.math.sqrt(tf.reduce_sum((handPos2 - obj_center[i])**2))
      #tf.print(handPos1)
      #tf.print(obj_center[i])
      if diss2 > 0.3:
        handLoss += diss2 * 1.5
    
  '''
  poseMean = 0.0
  
  #tf.print(part)
  part1 = tf.einsum("ab,cd->ad", part, tf.convert_to_tensor(loader.pose_cov_inv, dtype=tf.float32))
  
  part = tf.transpose(part, perm=[1, 0])
  
  part2 = tf.einsum("ab,cd->ad", part1, part)
  print(part2.shape)
  #exit()
  poseMean = tf.reduce_sum(part2)
  
  poseMean = tf.reduce_sum(part2)
  '''
  poseMean = 0.0
  pose_cov_inv = tf.convert_to_tensor(loader.pose_cov_inv, dtype=tf.float32)
  for i in range(batch_size):
    partThis = tf.reshape(part[i], (1,60))
    part1 = tf.tensordot(partThis, pose_cov_inv, axes=1)
    partThis = tf.transpose(partThis, perm=[1, 0])
    part2 = tf.tensordot(part1, partThis, axes=1)
    poseMean += tf.sqrt(part2)
  if poseMean < 900:
    poseMean = 0.0
  tf.print()
  tf.print(poseMean * 0.00005)
  tf.print(handLoss * 0.05)
  tf.print(mes)
  
  return poseMean * 0.00005+mes+handLoss * 0.1
  #return mes

def maskLabelLoss2(yTrue, yPred):
  zerosPattern = tf.zeros_like(yPred)
  mask = tf.math.abs(yTrue) > zerosPattern
  
  yTrue_m = tf.boolean_mask(yTrue, mask)
  yPred_m = tf.boolean_mask(yPred, mask)
  return tf.keras.losses.MSE(yTrue_m, yPred_m)


class TokenEmbeddingObj(layers.Layer):
  def __init__(self, maxlen, num_hid):
    super().__init__()
    self.denseEmb1 = tf.keras.layers.Dense(64)
    self.denseEmb2 = tf.keras.layers.Dense(64)
    self.denseEmb3 = tf.keras.layers.Dense(num_hid)
    self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

  def call(self, x):
    maxlen = x.shape[-2]
    x = self.denseEmb1(x)
    x = self.denseEmb2(x)
    x = self.denseEmb3(x)
    positions = tf.range(start=0, limit=maxlen, delta=1)
    positions = self.pos_emb(positions)
    result = x + positions
    return result

class TokenEmbeddingPpl(layers.Layer):
  def __init__(self, maxlen, num_hid):
    super().__init__()
    self.denseEmb1 = tf.keras.layers.Dense(128)
    self.denseEmb2 = tf.keras.layers.Dense(num_hid)
    self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

  def call(self, x):
    maxlen = x.shape[-2]
    x = self.denseEmb1(x)
    x = self.denseEmb2(x)
    positions = tf.range(start=0, limit=maxlen, delta=1)
    positions = self.pos_emb(positions)
    result = x + positions
    #result = x
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
    return tf.tile(mask, mult)

  def call(self, enc_out, target):
    input_shape = tf.shape(target)
    batch_size = input_shape[0]
    seq_len = input_shape[1]
    causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
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
      self.classifier = layers.Dense(num_classes)
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
    result = self.classifier(y)
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

  def generate(self, source, dec_start_input, dec_Pre= None):
    print("using oritnal generate step")
    source = source.reshape((1, self.target_maxlen, 36))
    enc = self.encoder(source)
    print("enter the loop")
    for i in range(self.target_maxlen - 1):
      dec_out = self.decode(enc, dec_start_input)
      #print("decode sccuess")
      #print(dec_out.shape)
      out = self.classifier(dec_out)
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
    return result
  
  def generate2(self, source, dec_start_input, dec_Pre= None):
    print("using oritnal generate step")
    source = source.reshape((1, self.target_maxlen, 36))
    enc = self.encoder(source)
    print("enter the loop")
    for i in range(self.target_maxlen - 1):
      dec_out = self.decode(enc, dec_start_input)
      out = self.classifier(dec_out)
      if i == self.target_maxlen - 2:
        result = tf.identity(out)
      out = out[:, -1, :]
      out = tf.expand_dims(out, axis=1)
      dec_start_input = tf.concat([dec_start_input, out], axis=-2)
    dec_start_input = tf.dtypes.cast(dec_start_input, tf.float32)
    dec2_input2 = tf.concat((tf.convert_to_tensor(dec_start_input, dtype=tf.float32), result[:,:-1,:]), axis=1)
    result = self([source, dec2_input2])
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
    obj_final_center = tf.tile(obj_final_center, (1,129,1))
    return obj_final_center

  def train_step(self, batch):
    """Processes one batch inside model.fit()."""
    print("using new train step")
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
      
      out = self.classifier(dec_out)
      
      if i == self.target_maxlen - 2:
        result = tf.identity(out)
      out = out[:, -1, :]
      out = tf.expand_dims(out, axis=1)
      dec2_input = tf.concat([dec2_input, out], axis=-2)
    #i = self.target_maxlen - 2
    dec_input = tf.dtypes.cast(dec_input, tf.float32)
    dec2_input2 = tf.concat((tf.expand_dims(dec_input[:,0,:], axis=1), result[:,:-1,:]), axis=1)
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
        self, num_hid=64, num_head=2, num_feed_forward=128, source_maxlen=60, target_maxlen=60, num_layers_enc=4, num_layers_dec=1, num_classes=10):
        super().__init__(num_hid, num_head, num_feed_forward, source_maxlen, target_maxlen, num_layers_enc, num_layers_dec, num_classes)
  
  def getObjCenterInfo(self, batch_size, source):
    obj_pos = tf.reshape(source, (batch_size, self.target_maxlen, 12,3))
    obj_final_pos = obj_pos[:, 0, :,:]
    obj_final_center = tf.reduce_mean(obj_final_pos, axis=-2)
    obj_final_center = tf.expand_dims(obj_final_center, axis=1)
    obj_final_center = tf.tile(obj_final_center, (1,129,1))
    return obj_final_center

  def train_step(self, batch):
    """Processes one batch inside model.fit()."""
    print("using new train step 2")
    source, target = batch
    dec_input = target[:, :-1, :]
    dec_input = tf.dtypes.cast(dec_input, tf.float32)
    dec_target = target[:, 1:, :]
    y = dec_target
    obj_center = self.getObjCenterInfo(batch_size, source)
    y = tf.concat((y, obj_center), axis = -1)
    with tf.GradientTape() as tape:
      y_pred1 = self([source, dec_input], training=True)
      loss1 = self.compiled_loss(y, y_pred1, regularization_losses=self.losses)
      new_input = tf.concat((tf.expand_dims(dec_input[:,0,:], axis=1), y_pred1[:,:-1,:]), axis=1)
      #with tf.GradientTape() as tape:
      y_pred2 = self([source, new_input], training=True)
      loss2 = self.compiled_loss(y, y_pred2, regularization_losses=self.losses)
      loss = loss1+loss2
      #loss = loss2
    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    #self.compiled_metrics.update_state(loss)
    self.compiled_metrics.update_state(y, y_pred2)
    #return {"loss": self.loss_metric.result()}
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
