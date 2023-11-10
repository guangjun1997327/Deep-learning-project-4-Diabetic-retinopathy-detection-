from tensorflow import keras
from keras import layers


INPUT_SIZE = 256
CLASS_NUM = 2
 # bn1 = tf.keras.layers.BatchNormalization (x)
# stage_name=2,3,4,5;  block_name=a,b,c
def CBlock(input_tensor, output_filter, stride, name):
    filter1, filter2 = output_filter
    #stride = 2
    x = layers.Conv2D(filter1, 3, strides=stride, padding='same', name='name'+name)(input_tensor)
    x = layers.BatchNormalization(name='bnc'+name)(x)
    x = layers.Activation('relu', name='resc'+name)(x)
    #stride=1
    x = layers.Conv2D(filter2, 3, strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization(name='bnc'+name+'_branch2b')(x)
    x = layers.Activation('relu', name='resc'+name+'_branch2b_relu')(x)
    #Identity block
    Identity = layers.Conv2D(filter2, 1, strides=stride, padding='same', name='resi'+name)(input_tensor)
    Identity = layers.BatchNormalization(name='bni'+name)(Identity)

    x = layers.add([x, Identity])
    x = layers.Activation('relu')(x)

    return x

def IBlock(input_tensor, output_filter, name):
    filter1, filter2 = output_filter

    x = layers.Conv2D(filter1, 3, strides=(1, 1), padding='same', name='res'+name+'_branch2a')(input_tensor)
    x = layers.BatchNormalization(name='bn'+name+'_branch2a')(x)
    x = layers.Activation('relu', name='res'+name+'_branch2a_relu')(x)

    x = layers.Conv2D(filter2, 3, strides=(1, 1), padding='same', name='res'+name+'_branch2b')(x)
    x = layers.BatchNormalization(name='bn'+name+'_branch2b')(x)
    x = layers.Activation('relu', name='res'+name+'_branch2b_relu')(x)

    Identity = input_tensor

    x = layers.add([x, Identity], name='res'+name)
    x = layers.Activation('relu', name='res'+name+'_relu')(x)

    return x

def Res18(input_shape, class_num,dropout_rate):
    input = keras.Input(shape=input_shape, name='input')

    # conv1
    x = layers.Conv2D(64, 7, strides=(2, 2), padding='same', name='conv1')(input)  # 7×7, 64, stride 2
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu', name='conv1_relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same', name='pool1')(x)   # 3×3 max pool, stride 2

    # conv2_x
    x = CBlock(input_tensor=x, output_filter=(64, 64), stride=(1, 1), name='2a')
    x = IBlock(input_tensor=x, output_filter=(64, 64), name='2b')

    # conv3_x
    x = CBlock(input_tensor=x, output_filter=(128, 128), stride=(2, 2), name='3a')
    x = IBlock(input_tensor=x, output_filter=(128, 128), name='3b')

    # conv4_x
    x = CBlock(input_tensor=x, output_filter=(256, 256), stride=(2, 2), name='4a')
    x = IBlock(input_tensor=x, output_filter=(256, 256), name='4b')

    # conv5_x
    x = CBlock(input_tensor=x, output_filter=(512, 512), stride=(2, 2), name='5a')
    x = IBlock(input_tensor=x, output_filter=(512, 512), name='5b')

    # average pool, 1000-d fc, softmax
    x = layers.AveragePooling2D((7, 7), strides=(1, 1), name='pool5')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(dropout_rate, name='dropout1')(x)
    x = layers.Dense(class_num, activation='softmax', name='fC1000')(x)

    model = keras.Model(input, x, name='res18')
    model.summary()
    return model

if __name__ == '__main__':
    model = Res18((INPUT_SIZE, INPUT_SIZE, 3), CLASS_NUM,0.5)
    print('Done.')
