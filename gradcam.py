import tensorflow as tf
import numpy as np
import cv2

LAST_CONV_LAYER = "conv2d_1"


def make_gradcam_heatmap(img_array, model):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(LAST_CONV_LAYER).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    heatmap /= tf.reduce_max(heatmap) + 1e-8

    print("Min:", np.min(heatmap.numpy()))
    print("Max:", np.max(heatmap.numpy()))
    print("Mean:", np.mean(heatmap.numpy()))

    return heatmap.numpy()

def overlay_heatmap(heatmap, image, alpha=0.25):
    
    threshold = np.percentile(heatmap, 95)
    heatmap[heatmap < threshold] = 0

    heatmap = cv2.resize(
        heatmap,
        (image.shape[1], image.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_TURBO
    )

    if len(image.shape) == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

    superimposed = cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0,
    )

    return superimposed