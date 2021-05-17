import tensorflow as tf
import numpy as np
import dataLoader

batch_size = 16

def customLoss(yTrue, yPred):
    loss_op = tf.reduce_sum(tf.losses.mean_squared_error(yTrue, yPred))
    return loss_op
def metricF(yTrue, yPred):
    loss_op = tf.reduce_sum(tf.losses.mean_squared_error(yTrue, yPred))
    return loss_op

human_inputs = tf.keras.Input(shape=(21,3))
obj_trajectory_all = tf.keras.Input(shape=(50,6))
obj_trajectory_current = tf.keras.Input(shape=(12))
human_encode1 = tf.keras.layers.Flatten()(human_inputs)
obj_encode1 = tf.keras.layers.Flatten()(obj_trajectory_all)
obj_encode1 = tf.keras.layers.Dense(128)(obj_encode1)
obj_encode1 = tf.keras.layers.Dense(64, activation='relu')(obj_encode1)
obj_encode1 = tf.keras.layers.Dense(32, activation='relu')(obj_encode1)
latent1 = tf.keras.layers.Concatenate(axis=-1)([human_encode1, obj_encode1])
latent = tf.keras.layers.Dense(128, activation='relu')(latent1)
latent = tf.keras.layers.Dense(128, activation='relu')(latent)
latent = tf.keras.layers.Dense(64, activation='relu')(latent)

obj_encode2 = tf.keras.layers.Flatten()(obj_trajectory_current)
obj_encode2 = tf.keras.layers.Dense(16)(obj_encode2)
obj_encode2 = tf.keras.layers.Dense(32, activation='relu')(obj_encode2)

decode = tf.keras.layers.Concatenate(axis=-1)([latent, obj_encode2])
decode = tf.keras.layers.Dense(128, activation='relu')(decode)
decode = tf.keras.layers.Dense(64)(decode)
decode = tf.keras.layers.Dense(63)(decode)
output = tf.keras.layers.Reshape((21,3))(decode)


model = tf.keras.Model(inputs=[human_inputs, obj_trajectory_all, obj_trajectory_current], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=customLoss, metrics=[metricF])
print(model.summary())

loader = dataLoader.trainDataLoader()
full_dataset = loader.getDataset()
full_dataset = full_dataset.shuffle(3)
val_dataset = full_dataset.take(100) 
train_dataset = full_dataset.skip(100)

train_dataset = train_dataset.batch(batch_size)
val_dataset = val_dataset.batch(1)

model.fit_generator(train_dataset, epochs = 30)

result = model.evaluate_generator(val_dataset, verbose=2)
print(result)