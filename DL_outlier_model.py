import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model

class ChannelAttention(Layer):
    def __init__(self, filters, ratio):
        super(ChannelAttention, self).__init__()
        self.filters = filters
        self.ratio = ratio

        def build(self, input_shape):
            self.shared_layer_one = Dense(self.filters // self.ratio,activation='relu', kernel_initializer='he_normal',
                                                          use_bias=True,
                                                          bias_initializer='zeros')
            self.shared_layer_two = Dense(self.filters, kernel_initializer='he_normal',
                                                          use_bias=True,
                                                          bias_initializer='zeros')

        def call(self, inputs):
            # AvgPool
            avg_pool = GlobalAveragePooling2D()(inputs)

            avg_pool = self.shared_layer_one(avg_pool)
            avg_pool = self.shared_layer_two(avg_pool)

            # MaxPool
            max_pool = GlobalMaxPooling2D()(inputs)
            max_pool = Reshape((1, 1, filters))(max_pool)

            max_pool = self.shared_layer_one(max_pool)
            max_pool = self.shared_layer_two(max_pool)

            attention = Add()([avg_pool, max_pool])
            attention = Activation('sigmoid')(attention)

            return Multiply()([inputs, attention])

class SpatialAttention(Layer):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        def build(self, input_shape):
            self.conv2d = Conv2D(filters=1,kernel_size=self.kernel_size,
                                                 strides=1,
                                                 padding='same',
                                                 activation='sigmoid',
                                                 kernel_initializer='he_normal',
                                                 use_bias=False)

        def call(self, inputs):
            # AvgPool
            avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(inputs)

            # MaxPool
            max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(inputs)

            attention = Concatenate(axis=3)([avg_pool, max_pool])

            attention = self.conv2d(attention)

            return Multiply([inputs, attention])

def Patchs_network(inputShape, pretrained_weights=None):
    # specify the inputs for the feature extractor network
    inputs1 = Input(inputShape)
    depth_conv1_1 = DepthwiseConv2D(3, activation='relu',strides=1, padding='same',depthwise_initializer='he_normal')(inputs1) # channel 3
    dep_bn1_1 = BatchNormalization()(depth_conv1_1)
    depth_conv1_2 = SeparableConv2D(12, 3, activation='relu',strides=2, padding='same',depthwise_initializer='he_normal',pointwise_initializer='he_normal')(dep_bn1_1) # channel 12
    dep_bn1_2 = BatchNormalization()(depth_conv1_2)
    maxpool1 = MaxPooling2D(pool_size=(3, 3),strides=1, padding='same')(dep_bn1_2) # channel 12

    inputs2 = Cropping2D((16,16))(inputs1)
    depth_conv2_1 = DepthwiseConv2D(3, activation='relu',strides=1, padding='same',depthwise_initializer='he_normal')(inputs2) # channel 3
    dep_bn2_1 = BatchNormalization()(depth_conv2_1)
    depth_conv2_2 = SeparableConv2D(12, 3, activation='relu', strides=1, padding='same',depthwise_initializer='he_normal',pointwise_initializer='he_normal')(dep_bn2_1)  # channel 12
    dep_bn2_2 = BatchNormalization()(depth_conv2_2)

    concat1 = Concatenate(axis=3)([maxpool1,dep_bn2_2]) # channel 24
    sp_conv1 = SeparableConv2D(28, 9, activation='relu',strides=1, padding='valid',depthwise_initializer='he_normal',
                               pointwise_initializer='he_normal')(concat1) # channel 28
    sp_bn1 = BatchNormalization()(sp_conv1)
    sp_conv2 = SeparableConv2D(32, 7, activation='relu',strides=1, padding='valid',depthwise_initializer='he_normal',
                               pointwise_initializer='he_normal')(sp_bn1) # channel 32
    sp_bn2 = BatchNormalization()(sp_conv2)
    maxpool2 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(sp_bn2)

    inputs3 = Cropping2D((24, 24))(inputs1)
    depth_conv3_1 = DepthwiseConv2D(3, activation='relu', strides=1, padding='same',depthwise_initializer='he_normal')(inputs3) # channel 27
    dep_bn3_1 = BatchNormalization()(depth_conv3_1)
    depth_conv3_2 = SeparableConv2D(16, 3, activation='relu', strides=1, padding='same',depthwise_initializer='he_normal',pointwise_initializer='he_normal')(dep_bn3_1)  # channel 32
    dep_bn3_2 = BatchNormalization()(depth_conv3_2)
    depth_conv3_3 = SeparableConv2D(32, 3, activation='relu', strides=1, padding='same',depthwise_initializer='he_normal',pointwise_initializer='he_normal')(dep_bn3_2)  # channel 32
    dep_bn3_3 = BatchNormalization()(depth_conv3_3)


    concat2 =Concatenate(axis=3)([maxpool2, dep_bn3_3]) # channel 64
    sp_conv3 = SeparableConv2D(96, 5, activation='relu', strides=1, padding='valid',depthwise_initializer='he_normal',
                               pointwise_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.01))(concat2)  # channel 96
    sp_bn3 = BatchNormalization()(sp_conv3)
    sp_conv4 = SeparableConv2D(128, 3, activation='relu', strides=1, padding='valid',depthwise_initializer='he_normal',
                               pointwise_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.01))(sp_bn3)  # channel 128
    sp_bn4 = BatchNormalization()(sp_conv4)
    CH_atten1 = ChannelAttention(128, 8)(sp_bn4)
    SP_atten1 = SpatialAttention(7)(CH_atten1)
    maxpool3 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(SP_atten1)

    inputs4 = Cropping2D((28, 28))(inputs1)
    depth_conv4_1 = DepthwiseConv2D(3, activation='relu', strides=1, padding='same',depthwise_initializer='he_normal')(inputs4)  # channel 96
    dep_bn4_1 = BatchNormalization()(depth_conv4_1)
    depth_conv4_2 = SeparableConv2D(32, 3, activation='relu', strides=1, padding='same',depthwise_initializer='he_normal',pointwise_initializer='he_normal')(dep_bn4_1)  # channel 128
    dep_bn4_2 = BatchNormalization()(depth_conv4_2)
    depth_conv4_3 = SeparableConv2D(64, 3, activation='relu', strides=1, padding='same',depthwise_initializer='he_normal',pointwise_initializer='he_normal')(dep_bn4_2)  # channel 128
    dep_bn4_3 = BatchNormalization()(depth_conv4_3)
    depth_conv4_4 = SeparableConv2D(128, 3, activation='relu', strides=1, padding='same',depthwise_initializer='he_normal',pointwise_initializer='he_normal')(dep_bn4_3)  # channel 128
    dep_bn4_4 = BatchNormalization()(depth_conv4_4)


    concat = Concatenate(axis=3)([maxpool3, dep_bn4_4]) # channel 256
    conv3 = SeparableConv2D(384, 3, activation='relu', strides=1, padding='valid',depthwise_initializer='he_normal',
                            pointwise_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.01))(concat)  # channel 384
    sp_bn5 = BatchNormalization()(conv3)
    CH_atten2 = ChannelAttention(384, 8)(sp_bn5)
    SP_atten2 = SpatialAttention(7)(CH_atten2)
    conv4 = SeparableConv2D(512, 3, activation='relu', strides=1, padding='valid',depthwise_initializer='he_normal',
                            pointwise_initializer='he_normal',kernel_regularizer=tf.keras.regularizers.L2(0.01))(SP_atten2)  # channel 512
    sp_bn6 = BatchNormalization()(conv4)
    CH_atten3 = ChannelAttention(512, 8)(sp_bn6)
    SP_atten3 = SpatialAttention(7)(CH_atten3)
    maxpool4 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(SP_atten3)

    outputs = Flatten()(maxpool4)

    # build the model
    model = Model(inputs=inputs1, outputs=outputs)
    model.summary()
    plot_model(model,to_file='Patch_net.png')

    if (pretrained_weights):
        model.load_weights(pretrained_weights)
    # return the model to the calling function
    return model
