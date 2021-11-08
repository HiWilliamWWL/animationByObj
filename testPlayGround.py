import tensorflow as tf
import numpy as np
a = np.array([[[1,2], [3,4]],[[2,4], [6, 8]],[[4,8], [12, 16]],[[8,16], [24, 32]]])
w = np.array([8, 4, 2, 1])
af = tf.convert_to_tensor(a, dtype=tf.float32)
wf = tf.convert_to_tensor(w, dtype=tf.float32)
result = tf.einsum("a,abc->bc", wf, af)
print(result)