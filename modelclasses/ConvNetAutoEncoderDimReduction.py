import tensorflow as tf
import tensorflow.keras.layers as layers

class EncoderDimReduction(tf.keras.Model):
    def __init__(self, l2_regularizer):
        super(EncoderDimReduction, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(100, (15, 15), activation='relu', padding='same',
                                            kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)
        self.maxp1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(75, (10, 10), activation='relu', padding='same',
                                            kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)
        self.maxp2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.flatten1 = tf.keras.layers.Flatten()
        self.encoded = tf.keras.layers.Dense(1024, activation='sigmoid')

    def call(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.flatten1(x)
        x = self.encoded(x)
        return x


class DecoderDimReduction(tf.keras.Model):
    def __init__(self, l2_regularizer):
        super(DecoderDimReduction, self).__init__()

        self.reshp1 = tf.keras.layers.Reshape((32, 32, 1))
        self.upsample00 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(75, (10, 10), activation='relu', padding='same',
                                            kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)
        self.upsample0 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv4 = tf.keras.layers.Conv2D(100, (15, 15), activation='relu', padding='same',
                                            kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)
        self.conv7 = tf.keras.layers.Conv2D(
            1, (15, 15), activation='sigmoid', padding='same')

    def call(self, x):
        x = self.reshp1(x)
        x = self.upsample00(x)
        x = self.conv3(x)
        x = self.upsample0(x)
        x = self.conv4(x)
        x = self.conv7(x)
        return x


class ConvNetAutoEncoderDimReduction(tf.keras.Model):
    def __init__(self, l2_regularizer, name=None):
        super(ConvNetAutoEncoderDimReduction, self).__init__(name=name)
        self.encoder = EncoderDimReduction(l2_regularizer)
        self.decoder = DecoderDimReduction(l2_regularizer)

    def call(self, x):
        x = self.encoder(x)
        x1 = self.decoder(x)
        return x, x1

