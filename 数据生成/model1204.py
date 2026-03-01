import tensorflow as tf
def creat_model():
    conv_base = tf.keras.applications.VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    #conv_base.summary()
    conv_base.get_layer('block5_conv3').output   #获取block5_conv3的输出
    sub_model = tf.keras.models.Model(inputs=conv_base.input, outputs=conv_base.get_layer('block5_conv3').output)  #设置模型的输入与输出
    #sub_model.summary()
    layer_names = ['block5_conv3', 'block4_conv3',  'block3_conv3', 'block5_pool']  #模型的最后四曾
     #最后四层，每层的输出
    layers_output = [conv_base.get_layer(layer_name).output for layer_name in layer_names]
    #创建一个多输出模型
    multi_out_model = tf.keras.models.Model(inputs=conv_base.input, outputs=layers_output)#
    multi_out_model.trainable = False
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    out_block5_conv3, out_block4_conv3, out_block3_conv3, out = multi_out_model(inputs)
    #print(out.shape)
    #print(out_block5_conv3.shape)
    #print(out_block4_conv3.shape)
    #print(out_block3_conv3.shape)
    x1 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=2, padding='same', activation='relu')(out)
    #print(x1.shape)
    x1 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x1)
    #print(x1.shape)
    x2 = tf.add(x1, out_block5_conv3)
    #print(x2.shape)
    #x2进行上采样(None, 14, 14, 512)
    #```bash
    x2 = tf.keras.layers.Conv2DTranspose(512, (3, 3), strides=2, padding='same', activation='relu')(x2)
    x2 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x2)
    x3 = tf.add(x2, out_block4_conv3)
    #print(x3.shape)
    #x3进行上采样(None, 28, 28, 512)

    #```bash
    x3 = tf.keras.layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation='relu')(x3)
    x3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x3)
    x4 = tf.add(x3, out_block3_conv3)
    #print(x4.shape)
    x5 = tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(x4)
    x5 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x5)
    #print(x5.shape)
    prediction = tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', activation='sigmoid')(x5)
    #print(prediction.shape)

    model = tf.keras.models.Model(inputs=inputs, outputs=prediction)
    return model