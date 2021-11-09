import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
#import ABOtransformer

#loader = ABOtransformer.loader
#batch_size = ABOtransformer.batch_size
loader = None
batch_size = None

def maskLabelLoss(yTrue, yPred):
  #hand 8,12 base-0
  print("enter loss")
  obj_center = None
  obj_center_allSeq = None
  in_train = False
  add_hand = True
  final_frame_hand = False
  if yTrue.shape[2] > 63:
    obj_center = yTrue[:, -1, -3:]
    obj_center_allSeq = yTrue[:, :, -3:]
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
  groupedPre = tf.reshape(tf.identity(yPred_m), (-1, 63))[:, 3:]
  part = groupedPre - tf.convert_to_tensor(loader.pose_mean, dtype=tf.float32)

  handLoss = 0.0
  mse_last = 0.0
  for i in range(batch_size):
    lastFrame = tf.cast( tf.reduce_sum(tf.cast(mask2[i], tf.float32)), tf.int32)
    lastMSE = tf.keras.losses.MSE(yTrue[i, lastFrame - 3:lastFrame - 1, :], yPred[i, lastFrame - 3:lastFrame - 1, :])
    mse_last += lastMSE * 10.0
  if in_train and add_hand:
    if final_frame_hand:
      for i in range(batch_size):
        lastFrame = tf.cast( tf.reduce_sum(tf.cast(mask2[i], tf.float32)), tf.int32)

        handPos1 = yPred[i, lastFrame-2, 8*3:8*3+3]
        handPos1 += yPred[i, lastFrame-2, :3]
        #handPos1 /= 5.0 #norm
        handPos1 = handPos1 * loader.ppl_std  + loader.ppl_mean
        
        diss1 = tf.math.sqrt(tf.reduce_sum((handPos1 - obj_center[i])**2))
        if diss1 > 0.2:
          handLoss += diss1 * 1.5
        handPos2 = yPred[i, lastFrame-2, 12*3:12*3+3]
        handPos2 += yPred[i, lastFrame-2, :3]
        #handPos2 /= 5.0 #norm
        handPos1 = handPos1 * loader.ppl_std  + loader.ppl_mean

        diss2 = tf.math.sqrt(tf.reduce_sum((handPos2 - obj_center[i])**2))
        #tf.print(handPos1)
        #tf.print(obj_center[i])
        if diss2 > 0.2:
          handLoss += diss2 * 1.5
    else:
      for i in range(batch_size):
        handPos1 = yPred[i, :, 8*3:8*3+3]
        handPos1 += yPred[i, :, :3]
        #handPos1 /= 5.0 #norm
        handPos1 = handPos1 * loader.ppl_std  + loader.ppl_mean
        
        diss1 = tf.math.sqrt(tf.reduce_sum((handPos1 - obj_center_allSeq[i])**2, axis = -1))
        diss1 = tf.boolean_mask(diss1, mask2[i])
        dissZero = tf.zeros_like(diss1)
        diss1 = tf.where(diss1 > 0.1, diss1*1.5, dissZero)
        handLoss += tf.reduce_sum(diss1)

        handPos2 = yPred[i, :, 12*3:12*3+3]
        handPos2 += yPred[i, :, :3]
        #handPos2 /= 5.0 #norm
        handPos2 = handPos2 * loader.ppl_std  + loader.ppl_mean

        diss2 = tf.math.sqrt(tf.reduce_sum((handPos2 - obj_center_allSeq[i])**2, axis = -1))
        diss2 = tf.boolean_mask(diss2, mask2[i])
        dissZero = tf.zeros_like(diss2)
        diss2 = tf.where(diss2 > 0.1, diss2*1.5, dissZero)
        handLoss += tf.reduce_sum(diss2)

    
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
  partThis = tf.reshape(part, (-1,60))
  part1 = tf.tensordot(partThis, pose_cov_inv, axes=1)
  '''
  partThis = tf.transpose(partThis, perm=[1, 0])
  part2 = tf.tensordot(part1, partThis, axes=1)
  part2 = tf.linalg.diag_part(part2)
  poseMean += tf.math.reduce_sum(tf.sqrt(part2))
  '''
  part2 = tf.math.multiply(part1, partThis)
  part2 = tf.math.reduce_sum(part2, axis=1)
  poseMean += tf.math.reduce_sum(tf.sqrt(part2))
  if poseMean < 300: #500
    poseMean = 0.0
  #tf.print()
  #tf.print("check loss")
  #tf.print(poseMean * 0.00005  500)
  #tf.print(handLoss * 0.001)
  #tf.print(mes)

  #tf.print(poseMean * 0.00005)
  #tf.print(mes)
  #tf.print()
  #tf.print(mse_last)
  
  return poseMean * 0.00001+mes+handLoss * 0.01+mse_last
  #return mes+handLoss * 0.01
  #return mes

def initialPoseLoss(yTrue, yPred):
    mes = tf.keras.losses.MSE(yTrue, yPred)
    poseMean = 0.0
    pose_cov_inv = tf.convert_to_tensor(loader.pose_cov_inv_init, dtype=tf.float32)
    partThis = tf.identity(yPred)[:, 3:] - tf.convert_to_tensor(loader.pose_mean_init, dtype=tf.float32)
    
    
    part1 = tf.tensordot(partThis, pose_cov_inv, axes=1)
    #tf.print(part1)
    #partThis = tf.transpose(partThis, perm=[1, 0])
    part2 = tf.math.multiply(part1, partThis)
    part2 = tf.math.reduce_sum(part2, axis=1)
    #tf.print(part2)
    
    poseMean += tf.math.reduce_sum(tf.sqrt(part2))
    if poseMean < 500: #500
        poseMean = 0.0
    #tf.print(poseMean * 0.0001)

    return poseMean * 0.0001 + mes * 1.3

def maskMSE(yTrue, yPred):
  if yTrue.shape[2] > 63:   #63
    yTrue = yTrue[:, :, :-3]
  zerosPattern = tf.zeros_like(yPred)
  mask = tf.cast(tf.math.abs(yTrue), tf.float32) > tf.cast(zerosPattern, tf.float32)
  
  yTrue_m = tf.boolean_mask(yTrue, mask)
  yPred_m = tf.boolean_mask(yPred, mask)
  return tf.keras.losses.MSE(yTrue_m, yPred_m)

def basicMSE(yTrue, yPred):
  mse = tf.keras.losses.MSE(yTrue, yPred)
  return mse

def basicMSE_metric(yTrue, yPred):
  mse = tf.keras.losses.MSE(yTrue, yPred)
  return mse

def maskLabelLoss_angles(yTrue, yPred):
  obj_center = None
  obj_center_allSeq = None
  in_train = False
  add_hand = True
  handLoss = 0.0
  if yTrue.shape[2] > loader.humanDimension:   #54
    obj_center = yTrue[:, -1, -3:]
    obj_center_allSeq = yTrue[:, :, -3:] * loader.obj_std + loader.obj_mean
    in_train = True
    yTrue = yTrue[:, :, :-3]
  zerosPattern = tf.zeros_like(yPred)
  mask = tf.cast(tf.math.abs(yTrue), tf.float32) > tf.cast(zerosPattern, tf.float32)
  
  yTrue_m = tf.boolean_mask(yTrue, mask)
  yPred_m = tf.boolean_mask(yPred, mask)

  zerosPattern = tf.zeros_like(yPred[:,:,0])
  mask = tf.reduce_sum(tf.math.abs(yTrue), axis=-1) > zerosPattern

  if in_train and add_hand:
    for i in range(batch_size):
      offsetIndex = 3 + loader.humanDimension_rot
      #handPos1 = yPred[i, :, offsetIndex + 8*3 : offsetIndex + 8*3+3]  #7
      handPos1 = tf.reshape(yPred[i, :, offsetIndex:], (-1, 20, 3))[:, 7]
      handPos1 += yPred[i, :, :3]
      #handPos1 /= 5.0 #norm
      handPos1 = handPos1 * loader.ppl_std_p  + loader.ppl_mean_p
      
      diss1 = tf.math.sqrt(tf.reduce_sum((handPos1 - obj_center_allSeq[i])**2, axis = -1))
      diss1 = tf.boolean_mask(diss1, mask[i])
      dissZero = tf.zeros_like(diss1)
      diss1 = tf.where(diss1 > 0.15, diss1*1., dissZero)
      handLoss += tf.reduce_sum(diss1)

      #handPos2 = yPred[i, :, offsetIndex + 12*3 : offsetIndex+12*3+3]   #11
      handPos2 = tf.reshape(yPred[i, :, offsetIndex:], (-1, 20, 3))[:, 11]   #11
      handPos2 += yPred[i, :, :3]
      #handPos2 /= 5.0 #norm
      handPos2 = handPos2 * loader.ppl_std_p  + loader.ppl_mean_p

      diss2 = tf.math.sqrt(tf.reduce_sum((handPos2 - obj_center_allSeq[i])**2, axis = -1))
      diss2 = tf.boolean_mask(diss2, mask[i])
      dissZero = tf.zeros_like(diss2)
      diss2 = tf.where(diss2 > 0.15, diss2*1., dissZero)
      handLoss += tf.reduce_sum(diss2)

  return 5.0 * tf.keras.losses.MSE(yTrue_m, yPred_m) + handLoss * 0.0001

def maskLabelLoss_angles_simple(yTrue, yPred):
  return tf.keras.losses.MSE(yTrue, yPred)

def maskMSE_angles(yTrue, yPred):
  if yTrue.shape[2] > loader.humanDimension:   #54
    yTrue = yTrue[:, :, :-3]
  zerosPattern = tf.zeros_like(yPred)
  mask = tf.cast(tf.math.abs(yTrue), tf.float32) > tf.cast(zerosPattern, tf.float32)
  
  yTrue_m = tf.boolean_mask(yTrue, mask)
  yPred_m = tf.boolean_mask(yPred, mask)
  return tf.keras.losses.MSE(yTrue_m, yPred_m)

def initialPoseLoss_angles(yTrue, yPred):
  mse1 = tf.keras.losses.MSE(yTrue[:, 3:-60], yPred[:, 3:-60])
  mse2 = tf.keras.losses.MSE(yTrue[:, -60:], yPred[:, -60:])
  #poseMean = 0.0
  '''
  pose_cov_inv = tf.convert_to_tensor(loader.pose_cov_inv_init, dtype=tf.float32)
  tf.print()
  #tf.print(pose_cov_inv)
  tf.print()
  #tf.print(tf.identity(yPred)[:, 3:])
  partThis_bN = tf.identity(yPred)[:, 3:] - tf.convert_to_tensor(loader.pose_mean_init, dtype=tf.float32)
  partThis_Nb = tf.transpose(partThis_bN, perm=[1, 0])
  tf.print(partThis_bN[0, :], summarize=51)
  tf.print()
  tf.print(pose_cov_inv[:, 1], summarize=51)
  part1 = tf.linalg.matmul(partThis_bN, pose_cov_inv)
  tf.print()
  tf.print(part1[0, :], summarize=51)
  tf.print()
  tf.print(partThis_Nb[:, 0], summarize=51)
  #partThis = tf.transpose(partThis, perm=[1, 0])
  #part2 = tf.math.multiply(part1, partThis_Nb)
  part2 = tf.linalg.matmul(part1, partThis_Nb)
  tf.print()
  tf.print(part2)
  #part2 = tf.math.reduce_sum(part2, axis=1)
  #tf.print(part2)
  poseMean = tf.math.reduce_sum(tf.sqrt(part2))
  tf.print(poseMean)
  if poseMean < 500: #500
      poseMean = 0.0
  '''
  #return 20.0*mse + poseMean * 0.001
  #tf.print(mse1)
  return 5.0 * (mse1 + mse2)