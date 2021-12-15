from tensorflow import keras
from tensorflow.keras import layers

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