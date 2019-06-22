import tensorflow as tf
from tensorflow.keras import layers
from res_block import ResBlock, ResBlock_Bottleneck

def ResNet_20():
    inputs = tf.keras.Input(shape = (256, 256, 3), name = "img")
    x = inputs

    x = layers.Conv2D(64, 7)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)
    for _  in range(3):
        x = ResBlock(num_feature_in = 64, num_feature_out = 64)(x)
    x = ResBlock(num_feature_in = 64, num_feature_out = 128, strides = (2,2)) (x)
    for _ in range(2):
        x = ResBlock(num_feature_in = 128, num_feature_out = 128)(x)
    x = ResBlock(num_feature_in = 128, num_feature_out = 256, strides = (2,2)) (x)
    for _ in range(2):
        x = ResBlock(num_feature_in = 256, num_feature_out = 256)(x)
    x = ResBlock(num_feature_in = 256, num_feature_out = 512, strides = (2,2))(x)
    for _ in range(2):
        x = ResBlock(num_feature_in = 512, num_feature_out = 512)(x)
    x = layers.AvgPool2D()(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(10, activation = "softmax")(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputs, name = "ResNet_20")
    model.compile(optimizer = tf.keras.optimizers.Adam(), loss = "categorical_crossentropy", metrics=["acc"])
    return model

if __name__ == "__main__":
    model  = ResNet_20()
    model.summary()
    tf.keras.utils.plot_model(model, "ResNet_20.png", show_shapes = True)
