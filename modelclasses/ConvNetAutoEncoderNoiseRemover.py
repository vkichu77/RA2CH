import tensorflow as tf
import tensorflow.keras.layers as layers

class EncoderNoiseRemover(tf.keras.Model):  # 100,10,50,7, 100,15,50,8
    def __init__(self, l2_regularizer):
        super(EncoderNoiseRemover, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(100, (10, 10), activation='relu', padding='same',
                                            kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)  # , input_shape=(128, 128, 1))
        self.maxp1 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(75, (7, 7), activation='relu', padding='same',
                                            kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)
        self.maxp2 = tf.keras.layers.MaxPooling2D((2, 2), padding='same')
        self.flatten1 = tf.keras.layers.Flatten()
        self.encoded = tf.keras.layers.Dense(1024, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.maxp2(x)
        x = self.flatten1(x)
        x = self.encoded(x)
        # print('enc '+x.shape)
        return x


class DecoderNoiseRemover(tf.keras.Model):
    def __init__(self, l2_regularizer):
        super(DecoderNoiseRemover, self).__init__()

        self.reshp1 = tf.keras.layers.Reshape((32, 32, 1))
        self.upsample00 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2DTranspose(
            75, (7, 7), activation='relu', padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)
        self.upsample0 = tf.keras.layers.UpSampling2D((2, 2))
        self.conv4 = tf.keras.layers.Conv2DTranspose(
            100, (10, 10), activation='relu', padding='same', kernel_initializer='glorot_uniform', kernel_regularizer=l2_regularizer)
        self.conv7 = tf.keras.layers.Conv2DTranspose(
            1, (10, 10), activation='sigmoid', padding='same', name='noiseDecoded')

    def call(self, x):
        x = self.reshp1(x)
        x = self.upsample00(x)
        x = self.conv3(x)
        # print(x.shape)
        x = self.upsample0(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = self.conv7(x)
        # print('tmp '+ x.shape)
        # print(x.shape)
        return x

class ConvNetAutoEncoderNoiseRemover(tf.keras.Model):
    def __init__(self, l2_regularizer, name=None):
        super(ConvNetAutoEncoderNoiseRemover, self).__init__(name=name)

        self.encoder = EncoderNoiseRemover(l2_regularizer)
        self.decoder = DecoderNoiseRemover(l2_regularizer)

    def call(self, x):
        x = self.decoder(self.encoder(x))
        return [], x




