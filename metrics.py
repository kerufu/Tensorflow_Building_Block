import tensorflow as tf
import numpy as np

class ClassificationMetricSet(tf.keras.metrics.Metric):
    def __init__(self, num_class, per_class=True, weighted=False):
        super(ClassificationMetricSet, self).__init__()
        self.e = 1e-31
        self.num_class = num_class
        self.per_class = per_class
        self.weighted = weighted

        self.tp = self.add_weight(
            shape=(self.num_class,),
            initializer='zeros',
            name='tp'
        )
        self.fp = self.add_weight(
            shape=(self.num_class,),
            initializer='zeros',
            name='fp'
        )
        self.tn = self.add_weight(
            shape=(self.num_class,),
            initializer='zeros',
            name='tn'
        )
        self.fn = self.add_weight(
            shape=(self.num_class,),
            initializer='zeros',
            name='fn'
        )
        self.label_count = self.add_weight(
            shape=(self.num_class,),
            initializer='zeros',
            name='label_count'
        )
        self.prediction_count = self.add_weight(
            shape=(self.num_class,),
            initializer='zeros',
            name='prediction_count'
        )

    def logit_to_one_hot(self, logit):
        return tf.one_hot(tf.argmax(logit, axis=1), depth=self.num_class)

    def reset_state(self):
        for v in self.variables:
            v.assign(tf.zeros(v.shape, dtype=v.dtype))

    def update_state(self, labels, predictions):
        label_index = tf.math.argmax(labels, axis=1)
        prediction_index = tf.math.argmax(predictions, axis=1)
        predictions = self.logit_to_one_hot(predictions)

        self.label_count.assign(self.label_count+tf.reduce_sum(labels, axis=0))
        self.prediction_count.assign(
            self.prediction_count+tf.reduce_sum(predictions, axis=0))

        match = tf.cast(tf.equal(label_index, prediction_index), tf.float32)
        match = tf.expand_dims(match, axis=-1)

        self.tp.assign(self.tp+tf.reduce_sum(match*labels, axis=0))
        self.tn.assign(self.tn+tf.reduce_sum(match *
                       np.ones((label_index.shape[0], self.num_class)), axis=0))
        self.tn.assign(self.tn-tf.reduce_sum(match*labels, axis=0))
        self.fp.assign(self.fp+tf.reduce_sum((1-match)*predictions, axis=0))
        self.fn.assign(self.fn+tf.reduce_sum((1-match)*labels, axis=0))

    def result(self):

        accuracy = (self.tp + self.tn) / tf.math.reduce_sum(self.label_count)
        precision = self.tp / (self.prediction_count + self.e)
        recall = self.tp / (self.label_count + self.e)
        f1_score = (2 * precision * recall) / (precision + recall + self.e)
        if self.per_class:
            return tf.math.reduce_mean(accuracy), precision, recall, f1_score
        else:
            if self.weighted:
                class_weights = tf.multiply(self.label_count, self.num_class) / tf.math.reduce_sum(self.label_count)
                accuracy *= class_weights
                precision *= class_weights
                recall *= class_weights
                f1_score *= class_weights
            return tf.math.reduce_mean(accuracy), tf.math.reduce_mean(precision), tf.math.reduce_mean(recall), tf.math.reduce_mean(f1_score)