import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = layers.MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_nested_unet(input_shape):
    inputs = layers.Input(input_shape)

    # Encoder
    e1, p1 = encoder_block(inputs, 64)
    e2, p2 = encoder_block(p1, 128)
    e3, p3 = encoder_block(p2, 256)
    e4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b1, e4, 512)
    d2 = decoder_block(d1, e3, 256)
    d3 = decoder_block(d2, e2, 128)
    d4 = decoder_block(d3, e1, 64)

    # Nested connections
    n1 = decoder_block(d1, e3, 256)
    n2 = decoder_block(n1, e2, 128)
    n3 = decoder_block(n2, e1, 64)

    n4 = decoder_block(d2, n2, 128)
    n5 = decoder_block(n4, e1, 64)

    n6 = decoder_block(d3, n5, 64)

    # Output
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(n6)

    model = Model(inputs, outputs, name="Nested_U-Net")
    return model

if __name__ == "__main__":
    input_shape = (256, 256, 1)  # Adjust based on your image dimensions
    model = build_nested_unet(input_shape)
    model.summary()