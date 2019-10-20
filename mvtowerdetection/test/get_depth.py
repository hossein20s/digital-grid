import os
import cv2
import numpy as np
from PIL import Image
from keras.engine.topology import Layer, InputSpec
import keras.utils.conv_utils as conv_utils
import tensorflow as tf
import keras.backend as K
import skimage.io as io


class DenseDepthAnalysis:
    def __init__(self, image_id, model, object_attr, width, height, depth_to_distance, min_depth=10, max_depth=1000, batch_size=1):
        self.object_attr = object_attr
        self.image_id = image_id
        self.model = model
        self.width = width
        self.height = height
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.depth_to_distance = depth_to_distance
    
    def rescale_img(self, imgs, width, height):
        return imgs.resize((width, height))

    def read_image(self):
        img = Image.open(os.path.join(os.getenv('TEST_IMAGES_PATH'), self.image_id+'.jpg'))
        return img

    def depth_norm(self, x, max_depth_):
        return max_depth_ / x

    def get_prediction(self):
        inp = self.read_image()
        img_width = np.asarray(inp).shape[0]
        img_height = np.asarray(inp).shape[1]

        rescaled_img = self.rescale_img(inp, self.width, self.height)
        inp = np.clip(np.asarray(rescaled_img, dtype=float) / 255, 0, 1)
        inp = np.expand_dims(inp, axis=0)
        
        predictions = self.model.predict(inp, batch_size=self.batch_size)

        outputs = np.clip(self.depth_norm(predictions, max_depth_=self.max_depth), self.min_depth, self.max_depth) / self.max_depth

        rescaled_output = cv2.resize(outputs.copy()[0], (img_height, img_width), interpolation=cv2.INTER_AREA)

        # io.imsave('test/results/xyz.png', rescaled_output)
        return rescaled_output

    def find_distance(self, depth):
        return int(depth * self.depth_to_distance)

    def revise_object_attr(self, depth_map):
        revised_object_attr = list()
        for attr in self.object_attr:
            object_depth = depth_map[attr[0]:attr[2], attr[1]:attr[3]].mean()
            if object_depth < int(os.getenv('DEPTH_THRESHOLD')):
                revised_object_attr.append(tuple((attr[0], attr[1], attr[2], attr[3], attr[4], self.find_distance(object_depth))))

        return revised_object_attr


class BilinearUpSampling2D(Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.data_format = K.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):
        input_shape = K.shape(inputs)
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        
        return tf.image.resize_images(inputs, [height, width], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))