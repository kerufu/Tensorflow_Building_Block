import tensorflow as tf

class WassersteinLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_true = y_true * 2 - 1
        return -tf.math.reduce_mean(y_true*y_pred)
    
class GradientPenalty():
    def call(self, discriminator, x_true, x_fake):
        epsilon = tf.random.uniform([x_true.shape[0]]+[1]*(len(x_true.shape)-1), 0.0, 1.0)
        x_mix = epsilon * tf.cast(x_true, tf.float32) + (1 - epsilon) * x_fake
        with tf.GradientTape() as tape:
            tape.watch(x_mix)
            discriminator_score_mix = discriminator(x_mix)
        gradients = tape.gradient(discriminator_score_mix, x_mix)
        gradient_mean = tf.sqrt(tf.reduce_sum(gradients**2, axis=list(range(1, len(gradients.shape)))))
        gradient_penalty = tf.reduce_mean((gradient_mean-1.0)**2)
        return gradient_penalty
    
class ScaleShiftInvariantLoss(tf.keras.losses.Loss):
    def __init__(self,):
        super(ScaleShiftInvariantLoss, self).__init__()

    def call(self, d, o):      
        n = tf.cast(tf.reduce_prod(o.shape[1:]), tf.float32)

        o = tf.reshape(o, [-1, n])
        d = tf.reshape(d, [-1, n])
        
        a = n * tf.cast(tf.reduce_sum(o*d, axis=1), tf.float32) - tf.cast(tf.reduce_sum(o, axis=1) * tf.reduce_sum(d, axis=1), tf.float32)
        a /= n * tf.reduce_sum(o*o, axis=1) - tf.cast(tf.reduce_sum(o, axis=1) * tf.reduce_sum(o, axis=1), tf.float32)
        b = tf.reduce_sum(d, axis=1) - a * tf.reduce_sum(o, axis=1)
        b /= n
        a = tf.expand_dims(a, 1)
        b = tf.expand_dims(b, 1)

        return tf.reduce_mean(tf.square(a*o+b-d))
