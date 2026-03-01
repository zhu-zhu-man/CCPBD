import tensorflow as tf

def creat_model():
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    # 下采样
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)  # 256*256*64

    x1 = tf.keras.layers.MaxPooling2D()(x)  # 128*128*64
    x1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)  # 128*128*128
    x1 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)  # 128*128*128

    x2 = tf.keras.layers.MaxPooling2D()(x1)  # 64*64*128
    x2 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)  # 64*64*256
    x2 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)  # 64*64*256

    x3 = tf.keras.layers.MaxPooling2D()(x2)  # 32*32*256
    x3 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)  # 32*32*512
    x3 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)  # 32*32*512

    x4 = tf.keras.layers.MaxPooling2D()(x3)  # 16*16*256
    x4 = tf.keras.layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)  # 16*16*1024
    x4 = tf.keras.layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)  # 16*16*1024

    # 上采样

    x5 = tf.keras.layers.Conv2DTranspose(512, (2, 2), padding="same", strides=2, activation='relu')(x4)  # 32*32*512
    x5 = tf.keras.layers.BatchNormalization()(x5)
    x6 = tf.concat([x5, x3], axis=-1)  # 32*32*1024

    x6 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x6)  # 32*32*512
    x6 = tf.keras.layers.BatchNormalization()(x6)
    x6 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x6)  # 32*32*512
    x6 = tf.keras.layers.BatchNormalization()(x6)

    x7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), padding="same", strides=2, activation='relu')(x6)  # 64*64*256
    x7 = tf.keras.layers.BatchNormalization()(x7)
    x8 = tf.concat([x7, x2], axis=-1)  # 64*64*512
    x8 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x8)  # 64*64*256
    x8 = tf.keras.layers.BatchNormalization()(x8)
    x8 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x8)  # 64*64*256
    x8 = tf.keras.layers.BatchNormalization()(x8)

    x9 = tf.keras.layers.Conv2DTranspose(128, (2, 2), padding="same", strides=2, activation='relu')(x8)  # 128*128*128
    x9 = tf.keras.layers.BatchNormalization()(x9)
    x10 = tf.concat([x9, x1], axis=-1)  # 128*128*256
    x10 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x10)  # 128*128*128
    x10 = tf.keras.layers.BatchNormalization()(x10)
    x10 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x10)  # 128*128*128
    x10 = tf.keras.layers.BatchNormalization()(x10)

    x11 = tf.keras.layers.Conv2DTranspose(64, (2, 2), padding="same", strides=2, activation='relu')(x10)  # 256*256*64
    x11 = tf.keras.layers.BatchNormalization()(x11)
    x12 = tf.concat([x11, x], axis=-1)  # 256*256*128
    x12 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x12)  # 256*256*64
    x12 = tf.keras.layers.BatchNormalization()(x12)
    x12 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x12)  # 256*256*64
    x12 = tf.keras.layers.BatchNormalization()(x12)

    output = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x12)  # 256*256*34

    return tf.keras.Model(inputs=inputs, outputs=output)
model = creat_model()
model.summary()