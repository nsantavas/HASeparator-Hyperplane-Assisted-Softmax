import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend

weight_decay = 1E-4


class HASeparator(tf.keras.layers.Layer):
    """Custom layer based on  Hyperplane-Assisted Softmax separator paper:
    
    I. Kansizoglou, N. Santavas, L. Bampis and A. Gasteratos, "HASeparator: Hyperplane-Assisted Softmax," 
    2020 19th IEEE International Conference on Machine Learning and Applications (ICMLA), 
    2020, pp. 519-526, doi: 10.1109/ICMLA51294.2020.00087.

    """    
    def __init__(self, num_classes:int, margin: float, scale: float, **kwargs):
        """
        Args:
            num_classes (int): Total number of classes
            margin (float): Margin value, above of which extra penalty will be added
            scale (float): Scaler value by which logits space will be multiplied
        """   
        super(HASeparator, self).__init__(**kwargs)   
        self.num_classes = num_classes
        self.logist_scale = scale
        self.margin = margin

    def build(self, input_shape):

        self.w = self.add_variable(
            "weights",
            trainable=True,
            shape=[int(input_shape[-1]), self.num_classes])
        self.inputShape = int(input_shape[-1])

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        labels = tf.cast(labels, tf.int32)
        gr_w = tf.expand_dims(tf.gather(tf.transpose(normed_w), labels,
                                        axis=0),
                              axis=-1)
        temp = tf.tile(tf.expand_dims(normed_w, 0), [tf.shape(embds)[0], 1, 1])
        dw = tf.subtract(gr_w, temp)
        normed_dw = tf.nn.l2_normalize(dw, axis=1, name='normed_dw')
        win = tf.einsum('...ij,...ijl->...il', normed_embds, normed_dw)
        penalties = tf.math.subtract(self.margin,
                                     tf.math.minimum(self.margin, win))

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        logits = tf.scalar_mul(self.logist_scale, cos_t, 'scaled_logist')
        logits = tf.nn.softmax(logits)

        return logits, penalties