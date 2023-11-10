img_path = 'IDRID_dataset/images/train/IDRiD_010.jpg'

def pretrained_path_to_tensor(img_path):
    #read the picture and change it to array
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.crop_to_bounding_box(img,0,200,2848,3550)
    img = tf.image.pad_to_bounding_box(img, offset_height=400, offset_width=0, target_height=3550, target_width=3550)
    img = tf.image.resize(img,[256,256])
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    print(img)
    return img
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    # 首先，我们创建一个模型，将输入图像映射到最后一个conv层的激活以及输出预测
    #create a model; reflect the model to the last conv activation or ouputs layers
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    #然后，我们为输入图像计算top预测类关于最后一个conv层的激活的梯度
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        #如果没有传入pred_index,就计算pred[0]中最大的值对应的下标号index
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)这是输出神经元(预测概率最高的或者选定的那个)对最后一个卷积层输出特征图的梯度
    # with regard to the output feature map of the last conv layer
    # grads.shape(1, 10, 10, 2048)
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient这是一个向量,每一项都是 指定特征图通道上的平均值
    # over a specific feature map channel
    # pooled_grads 是一个一维向量,shape=(2048,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
   # pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    # last_conv_layer_output[0]是一个三维的卷积层 ,@矩阵相乘(点积)
    #last_conv_layer_output.shape  =(10, 10, 2048)
    last_conv_layer_output = last_conv_layer_output[0]
    #heatmap (10, 10, 1) = (10, 10, 2048)  @(2048,)相当于(10, 10, 2048)乘以(2048,1)
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    # tf.squeeze 去除1的维度,(10, 10)
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    # tf.maximum(heatmap, 0) 和0比较大小,返回一个>=0的值,相当于relu,然后除以heatmap中最大的 值,进行normalize归一化到0-1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    # last_conv_layer_output[0]是一个三维的卷