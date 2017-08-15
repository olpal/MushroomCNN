import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import cnn as cn
import parameters as hp
import sys
import util as u
import csv
import numpy as np

version=1.0

u.printf ("Creating model parameters")
train_params = hp.ModelParameters()
exec_params = hp.ExecutionParameters()

u.printf ("Initializing Neural Network parameters...")
x = tf.placeholder('float32', [None, (train_params.image_size * train_params.image_size)])
keep_prob = tf.placeholder(tf.float32)

"""Assign passed in variables if they exist"""
if len(sys.argv) > 1:
    u.printf ("Arguments found, loading...")
    exec_params.model_directory = sys.argv[1]
    exec_params.input_image = sys.argv[2]

def load_model():
    parameters=[]
    exec_params.variable_full_path='{}/{}'.format(exec_params.model_directory,train_params.saved_variable_file_name)
    exec_params.model_full_path = '{}/{}'.format(exec_params.model_directory, train_params.saved_model_file_name)
    try:
        with open(exec_params.variable_full_path, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                parameters.append(row)
            train_params.number_of_classes = int(parameters[0][0])
            train_params.neuron_multiplier = int(parameters[0][1])
            train_params.convolutional_filter = int(parameters[0][2])
            train_params.convolutional_layer_count = int(parameters[0][3])
            return True
    except:
        print "No file found at location {}".format(exec_params.variable_full_path)
        return False

def create_network():
    """Neural Network Variables"""
    cnn = cn.CNN(x, keep_prob, train_params.convolutional_layer_count, train_params.image_size, train_params.number_of_classes,
                 train_params.neuron_multiplier, train_params.convolutional_filter)
    prediction = cnn.return_network()
    return tf.round(tf.nn.sigmoid(prediction))

def segment_image(image):
    slices = []
    shape = image.shape
    yStartPos = 0
    finalY = False
    while True:
        finalX = False
        xStartPos = 0
        while True:
            slices.append((yStartPos, (yStartPos + exec_params.y_size), xStartPos, (xStartPos + exec_params.x_size)))
            xStartPos = (xStartPos + (exec_params.x_size - exec_params.x_overlap))
            if (xStartPos + exec_params.x_size) > shape[1] and finalX:
                break
            elif (xStartPos + exec_params.x_size) > shape[1]:
                xStartPos = (shape[1] - exec_params.x_size)
                finalX = True
        yStartPos = (yStartPos + (exec_params.y_size - exec_params.y_overlap))
        if (yStartPos + exec_params.y_size) > shape[0] and finalY:
            break
        elif (yStartPos + exec_params.y_size) > shape[0]:
            yStartPos = (shape[0] - exec_params.y_size)
            finalY = True

    return slices


def stitch_image(slices, predictions, shape):
    predicted_image = np.zeros((shape[0],shape[1]),dtype=int)
    image_position=0
    for slice in slices:
        predicted_image[slice[0]:slice[1], slice[2]:slice[3]] = np.reshape(predictions[image_position],(train_params.image_size,train_params.image_size))
        image_position+=1
    return predicted_image


def predict_images(sess, network, image, slices):
    predicted_images=[]
    for slice in slices:
        to_process=[]
        to_process_image = image[slice[0]:slice[1], slice[2]:slice[3]]
        to_process.append(np.reshape(to_process_image, (train_params.image_size*train_params.image_size,)))
        predicted_images.append(sess.run([network], feed_dict={x: to_process, keep_prob: 1.}))
    return predicted_images

def export_data(image_data, shape):
    np.set_printoptions(threshold=np.inf)
    image_file = open('{}/predicted_data.txt'.format(exec_params.model_directory), 'w')

    reshaped_prediction = np.reshape(image_data, (shape[0] * shape[1],))
    reshaped_prediction = reshaped_prediction.astype(int)

    image_file.write("{}\n".format((np.array_str(reshaped_prediction, max_line_width=1000000))))
    image_file.close()

def export_image(image_data):
    plt.imshow(image_data, cmap="gray")
    plt.imsave('{}/predicted_image.png'.format(exec_params.model_directory), image_data, cmap="gray", dpi=250)

if __name__ == '__main__':
    u.printf("Executing version {} of the code".format(version))
    if not load_model():
        exit()
    network = create_network()
    image = plt.imread(exec_params.input_image)
    slices = segment_image(image)

    u.printf("Predicting image...")
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, exec_params.model_full_path)
        predicted_images=predict_images(sess, network, image, slices)

    u.printf("Stitching image")
    full_predicted_image = stitch_image(slices,predicted_images,image.shape)
    if exec_params.export_data:
        u.printf("Exporting data")
        export_data(full_predicted_image, image.shape)
    if exec_params.export_image:
        u.printf("Exporting image")
        export_image(full_predicted_image)