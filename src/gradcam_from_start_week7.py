"""Week 7 Grad-CAM utilities for freshly trained model."""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

def find_last_conv_layer(model_obj):
    for layer in reversed(model_obj.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name
        if isinstance(layer, keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, keras.layers.Conv2D):
                    return sub_layer.name
    raise ValueError("No Conv2D layer found.")

def make_gradcam_heatmap(image_array, model_obj, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model_obj)

    nested_backbone = None
    for layer in model_obj.layers:
        if isinstance(layer, keras.Model):
            try:
                layer.get_layer(last_conv_layer_name)
                nested_backbone = layer
                break
            except Exception:
                pass

    if nested_backbone is not None:
        conv_layer = nested_backbone.get_layer(last_conv_layer_name)
    else:
        conv_layer = model_obj.get_layer(last_conv_layer_name)

    grad_model = keras.Model(inputs=model_obj.inputs, outputs=[conv_layer.output, model_obj.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    if max_value == 0:
        return np.zeros_like(heatmap.numpy())

    return (heatmap / max_value).numpy()

def overlay_heatmap(original_rgb, heatmap, alpha=0.40):
    heatmap_resized = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_rgb, 1 - alpha, heatmap_color, alpha, 0)
