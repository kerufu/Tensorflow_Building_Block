import tensorflow as tf

class WassersteinLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true = y_true * 2 - 1
        return -tf.math.reduce_mean(y_true*y_pred)
    
class GradientPenalty():
    def call(self, discriminator, x_true, x_fake):
        epsilon = tf.random.uniform([x_true.shape[0]]+[1]*(len(x_true.shape)-1), 0.0, 1.0)
        x_mix = epsilon * tf.cast(x_true, tf.float32) + (1 - epsilon) * x_fake
        with tf.GradientTape() as tape:
            tape.watch(x_mix)
            discriminaotr_score_mix = discriminator(x_mix)
        gradients = tape.gradient(discriminaotr_score_mix, x_mix)
        gradient_mean = tf.sqrt(tf.reduce_sum(gradients**2, axis=list(range(1, len(gradients.shape)))))
        gradient_penalty = tf.reduce_mean((gradient_mean-1.0)**2)
        return gradient_penalty
