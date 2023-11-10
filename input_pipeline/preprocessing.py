import gin
import tensorflow as tf

@gin.configurable
def dataload(filepath,labelpath):
  imgset = []
  data = pd.read_csv(labelpath)
  len = data["Image name"].shape[0]
  # label
  labelset =np.asarray (data["Retinopathy grade"])
  labelset[labelset<2] = 0
  labelset[labelset>=2] = 1
  #image
  for i in range(0,len):
    name = data["Image name"][i]
    impath = os.path.join(filepath,name+'.jpg')
    #image processing
    # img = Image.open(impath)
    # cropim = img.crop((200,0,3750,2848))
    # img = cropim.resize((256,256))
    # img = np.asarray(img)
    # tf image processing
    img = tf.io.read_file(impath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.crop_to_bounding_box(img,0,200,2848,3550)
    img = tf.image.pad_to_bounding_box(img, offset_height=400, offset_width=0, target_height=3550, target_width=3550)
    img = tf.image.resize(img,[256,256])
    imgset.append(img)
  dataset = tf.data.Dataset.from_tensor_slices((imgset, labelset))
  return dataset
def preprocess(img, label, img_height, img_width):
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.crop_to_bounding_box(img, 0, 200, 2848, 3550)
    img = tf.image.pad_to_bounding_box(img, offset_height=400, offset_width=0, target_height=3550, target_width=3550)
    img = tf.image.resize(img, [256, 256])
    return img, label
def augment(image, label):
    """Data augmentation"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label
traindsag = trainds.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)