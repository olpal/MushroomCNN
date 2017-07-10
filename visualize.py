import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def visualize_map(sess, layer_map, convolutional_layer, test_images, test_labels, number_of_images, data, labels,
                  predictions, image_size, parameters):

    if number_of_images > len(test_labels):
        number_of_images = len(test_labels)

    for position in range(number_of_images):
        image = test_images[position:position + 1]
        label = test_labels[position:position + 1]
        label_scalar = np.where(label == 1)[1]
        convolutional_value, output_value = sess.run([convolutional_layer, predictions], feed_dict={data: image})

        max_value = np.max(output_value)
        predicted_scalar = np.where(output_value == max_value)[1]

        map_response = sess.run(layer_map, feed_dict={labels: label, convolutional_layer: convolutional_value})

        map_visualized = list(map(lambda x: ((x - x.min()) / (x.max() - x.min())), map_response))

        for visual, original in zip(map_visualized, image):
            plt.imshow(1 - np.resize(original, [image_size, image_size]), cmap='gray_r')
            plt.imshow(visual, cmap='jet', alpha=0.35, interpolation='none', vmin=0, vmax=1)
            cmap_file = '{}/map_{}_{}_{}.{}'.format(parameters.output_dir, position, label_scalar, predicted_scalar,
                                                    parameters.output_file_ext)
            plt.savefig(cmap_file)
            plt.close()


def get_map(label, convolutional_layer, image_size, fully_connected_layer):
    output_channels = int(convolutional_layer.get_shape()[-1])

    convolutional_resized = tf.image.resize_bilinear(convolutional_layer, [image_size, image_size])

    with tf.variable_scope('', reuse=True):
        label_w = tf.gather((fully_connected_layer), label)

        label_w = tf.reshape(label_w, [-1, output_channels, 1])
        label_w = tf.nn.relu(label_w)
        label_w = tf.reduce_sum(label_w, 0, keep_dims=True)

        convolutional_resized = tf.reshape(convolutional_resized, [-1, image_size * image_size, output_channels])

    classmap = tf.matmul(convolutional_resized, label_w)
    classmap = tf.reshape(classmap, [-1, image_size, image_size])
    return classmap


def plot_model_training(training_values, validation_values, data_label, parameters):
    plt.title("Model {} Graph".format(data_label))
    plt.plot(zip(*training_values)[1], zip(*training_values)[0], '-b', label="Training data", ls=':')
    plt.plot(zip(*validation_values)[1], zip(*validation_values)[0], '-r', label="Validation data", ls='--')
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("{}".format(data_label))
    plt.savefig("{}/{}.{}".format(parameters.output_dir,data_label,parameters.output_graph_ext))
    plt.clf()
