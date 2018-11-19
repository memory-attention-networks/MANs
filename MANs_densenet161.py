# -*- coding: utf-8 -*-

from keras.optimizers import SGD
from keras.layers import Input, merge, ZeroPadding2D,Cropping2D
from keras.layers.core import Dense, Dropout, Activation,Reshape,Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D,GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.layers import concatenate,multiply
from sklearn.metrics import log_loss
from keras.layers import add,LSTM,Bidirectional,GRU
from custom_layers.scale_layer import Scale
from keras.regularizers import l2,l1
import tensorflow as tf

reg=l2(1e-4)
num_frame=224
feature_dim=224
alpha=16

def MANs_model():
    '''

    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5
    nb_dense_block = 4
    growth_rate = 48

    reduction = 0.5
    dropout_rate = 0.0
    weight_decay = 1e-4
    compression = 1.0 - reduction


    global concat_axis
    if K.image_dim_ordering() == 'tf':
      concat_axis = 3
      img_input = Input(shape=(224, 224, 3), name='data')
    else:
      concat_axis = 1
      img_input = Input(shape=(3, 224, 224), name='data')



    nb_filter = 96
    nb_layers = [6,12,36,24] # For DenseNet-161
    nb_lstm=128    # For TARM

    ## Temporal Attention Recalibration Module(TARM) for three channel coordinate feature
    img_channel1=Lambda(lambda x: x[:,:,:,0])(img_input)
    img_reshape=Reshape((num_frame,feature_dim))(img_channel1)
    img_dense=Dense(nb_lstm)(img_reshape)
    #obtain the memory information
    lstm_output = Bidirectional(GRU(nb_lstm, kernel_initializer='orthogonal',recurrent_initializer='orthogonal',dropout=0,recurrent_dropout=0,
                             kernel_regularizer=reg,return_sequences=True), merge_mode='sum')(img_dense)
    lstm_weight=Lambda(attention,arguments={'nb_lstm':nb_lstm},output_shape=[num_frame,nb_lstm])(img_dense)
    lstm_output=multiply([lstm_output,lstm_weight])
    lstm=Dense(feature_dim)(lstm_output)
    lstm1=add([img_reshape,lstm])  # a residual module, adding by the memory information and temporal attention recalibration information
    
    ## channel 2
    img_channel2=Lambda(lambda x: x[:,:,:,1])(img_input)
    img_reshape=Reshape((num_frame,feature_dim))(img_channel2)
    img_dense=Dense(nb_lstm)(img_reshape)
    lstm_output = Bidirectional(GRU(nb_lstm, kernel_initializer='orthogonal',recurrent_initializer='orthogonal',dropout=0,recurrent_dropout=0,
                             kernel_regularizer=reg,return_sequences=True), merge_mode='sum')(img_dense)
    lstm_weight=Lambda(attention,arguments={'nb_lstm':nb_lstm},output_shape=[num_frame,nb_lstm])(img_dense)
    lstm_output=multiply([lstm_output,lstm_weight])
    lstm=Dense(feature_dim)(lstm_output)
    lstm2=add([img_reshape,lstm])

    ## channel 3
    img_channel3=Lambda(lambda x: x[:,:,:,2])(img_input)
    img_reshape=Reshape((num_frame,feature_dim))(img_channel3)
    img_dense=Dense(nb_lstm)(img_reshape)
    lstm_output = Bidirectional(GRU(nb_lstm, kernel_initializer='orthogonal',recurrent_initializer='orthogonal',dropout=0,recurrent_dropout=0,
                             kernel_regularizer=reg,return_sequences=True), merge_mode='sum')(img_dense)
    lstm_weight=Lambda(attention,arguments={'nb_lstm':nb_lstm},output_shape=[num_frame,nb_lstm])(img_dense)
    lstm_output=multiply([lstm_output,lstm_weight])
    lstm=Dense(feature_dim)(lstm_output)
    lstm3=add([img_reshape,lstm])

    lstm1=Reshape((224,224,1))(lstm1)
    lstm2=Reshape((224,224,1))(lstm2)
    lstm3=Reshape((224,224,1))(lstm3)
    x=concatenate([lstm1,lstm2,lstm3],axis=-1)


    ## Spatial Convolution Module(SCM)
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(x)
    x = Conv2D(nb_filter, 7, 7, subsample=(2, 2), name='conv1', bias=False,kernel_regularizer=reg)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)

    x_new = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
    x_new = Dense(60, name='fc6_new')(x_new)
    x_new = Activation('softmax', name='prob_new')(x_new)
    model = Model(img_input, x_new)
    model.load_weights('densenet161_weights_tf.h5',by_name='true')

    return model

def duplicate(x,nb_lstm):
    '''
    repeat the vector
    '''
    y=x
    for i in range(nb_lstm-1):
        y=concatenate([y,x])
    return y

def attention(x_lstm,nb_lstm):
    '''
    # Arguments
        x_lstm: the input coordinate feature
        nb_lstm: number of lstm neuron
    # Return
        attention_weight:the frame-wise attention weight
    '''
    x_lstm1=tf.transpose(x_lstm,[0,2,1])
    ap1=GlobalAveragePooling1D()(x_lstm1)

    ap1=Reshape((num_frame,1))(ap1)
    ap2=Lambda(duplicate,output_shape=[num_frame,nb_lstm],arguments={'nb_lstm':nb_lstm})(ap1)
    lstm1=tf.transpose(ap2,[0,2,1])
    
    nb_fc = int(num_frame / alpha)
    fc1 = Dense(nb_fc,activation='relu')(lstm1)

    fc2=Dense(num_frame,activation='sigmoid')(fc1)
    attention_weight=tf.transpose(fc2,[0,2,1])

    return attention_weight

def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor 
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Conv2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False,kernel_regularizer=reg)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Conv2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False,kernel_regularizer=reg)(x)


    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout 
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False,kernel_regularizer=reg)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis, name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter

if __name__ == '__main__':

    model=MANs_model()
