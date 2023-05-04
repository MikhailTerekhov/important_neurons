import tensorflow as tf
import PIL
from io import BytesIO
import sys
import urllib
from tensorflow.keras.preprocessing import image
import tensorflow_addons as tfa
import tqdm
import numpy as np

import lucid.optvis.param as param
import lucid.optvis.objectives as objectives
import lucid.modelzoo.vision_models as models
import lucid.optvis.render as render

tf.compat.v1.disable_eager_execution()

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

model = models.InceptionV1()
print(model.layers)
for _ in range(10):
    ind = np.random.randint(0, 800)
    print(ind)
    obj = objectives.channel("mixed4e", ind) - 1e2 * objectives.diversity("mixed4e")
    print(f'{obj.description}')
    # print(f'{obj.objective_func("mixed4e")}')
    p = lambda: param.image(256, batch=4)
    imgs = render.render_vis(model, obj, p, verbose=False)
    imgs = (imgs[0] * 255).astype(np.uint8)
    zeros = np.zeros((256, 20, 3), dtype=np.uint8)
    imgs = sum([[imgs[i], zeros] for i in range(imgs.shape[0] - 1)], []) + [imgs[imgs.shape[0] - 1, ...]]
    concat = np.concatenate(imgs, axis=1)

    print(f'{concat.shape=}')
    # Concatenate horizontally the images from imgs and display them
    PIL.Image.fromarray(concat).show()

sys.exit(1)

def load_image(url):
    with urllib.request.urlopen(url) as url:
        img = image.load_img(BytesIO(url.read()), target_size=(224, 224))
    return image.img_to_array(img)

# Load vgg19 from keras
vgg19 = tf.keras.applications.vgg19.VGG19(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
# Print summary
# vgg19.summary()
# Load image from the web given its address, preprocess it to be compatible with vgg19, and predict its class
def load_and_predict(url):
    img = load_image(url)
    # Display the image
    # PIL.Image.fromarray(img.astype('uint8')).show()

    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)
    labels = vgg19.predict(img)
    print('Predicted:', tf.keras.applications.vgg19.decode_predictions(labels, top=3)[0])


# do a deepdream visualization for a neuron in the given layer
def visualize(layer_name, filter_index):
    # Create a model that maps the input image to the activations of the given layer
    layer = vgg19.get_layer(layer_name)
    feature_extractor = tf.keras.Model(inputs=vgg19.inputs, outputs=layer.output)
    # Create a loss function that maximizes the activation
    # of the nth filter of the layer considered
    def loss(input_image):
        # rotate the image by 5 degrees, crop what is outside
        # the image, and normalize it
        deg = tf.random.uniform(shape=[], minval=-5, maxval=5)
        noise = tf.random.normal(shape=tf.shape(input_image), mean=0, stddev=0.05)
        augmented = tfa.image.rotate(input_image, deg, interpolation='BILINEAR', fill_mode='wrap') + noise

        activation = feature_extractor(augmented)
        return -activation[:, :, :, filter_index]
    # Create a gradient ascent optimizer
    # initialized with the input image
    input_image = tf.Variable(tf.random.uniform(shape=(1, 224, 224, 3)))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    # Run the optimization for 100 steps
    # print(f'{loss(input_image)[:5,:5,:1]=}')
    for _ in tqdm.trange(1000):
        optimizer.minimize(lambda: loss(input_image), var_list=[input_image])
    print(f'After: {tf.reduce_sum(loss(input_image))=}')
    # Display the image
    PIL.Image.fromarray(tf.cast(input_image * 255, tf.uint8).eval()[0]).show()

# load_and_predict('https://upload.wikimedia.org/wikipedia/commons/6/6e/Golde33443.jpg')
visualize('block5_conv2', 0)