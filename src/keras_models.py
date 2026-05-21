from keras.layers import Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, Input
from keras.models import Model

# Keras Model
def ResNet18(input_shape=(48, 48, 1), n_emotions=7, filters_levels=[64, 128, 256, 512], kernel_size=(3, 3)):
    
    def residual_block(x, filters, stride=1):
        shortcut = x
        
        x = Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        x = Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same', use_bias=False)(shortcut)
            shortcut = BatchNormalization()(shortcut)
            
        x = Add()([x, shortcut])
        x = ReLU()(x)
        return x

    inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size, strides=1, padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = residual_block(x, filters_levels[0], stride=1)
    x = residual_block(x, filters_levels[0], stride=1)

    x = residual_block(x, filters_levels[1], stride=2)
    x = residual_block(x, filters_levels[1], stride=1)

    x = residual_block(x, filters_levels[2], stride=2)
    x = residual_block(x, filters_levels[2], stride=1)

    x = residual_block(x, filters_levels[3], stride=2)
    x = residual_block(x, filters_levels[3], stride=1)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(n_emotions, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='ResNet18')
    return model
