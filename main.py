import datetime
import tensorflow as tf
import cnn as cn
import visualize as vs
import dataset as ds
import parameters as hp
import sys
import util as u
import datetime as dt

u.printf ("Starting model execution at {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


"""Variables"""
u.printf ("Intializing variables...")
training_losses = []
training_accuracies = []
validation_losses = []
validation_accuracies = []
dataset_location = "/Users/aolpin/Documents/School/thesis/testdata/data"
output_dir = "/Users/aolpin/Documents/School/thesis/results/{}".format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
training_batch_size = 100
learning_rate=0.001
dropout = 0.8
epochs=2000
convolutional_layer_count = 2
version=2.4
neuron_multiplier=40


"""Assign passed in variables if they exist"""
if len(sys.argv) > 1:
    u.printf ("Arguments found, looding...")
    dataset_location = sys.argv[1]
    output_dir = sys.argv[2]
    training_batch_size = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    dropout = float(sys.argv[5])
    epochs = int(sys.argv[6])
    convolutional_layer_count = int(sys.argv[7])
    neuron_multiplier = int(sys.argv[8])

"""Data sets"""
u.printf ("Building dataset...")
data_set = ds.DataSet(dataset_location, output_dir)
test_labels, test_data, validation_labels, validation_data, train_labels, train_data = data_set.load_datasets()
#test_labels_real, test_data_real, validation_labels_real, validation_data_real, train_labels_real, train_data_real = data_set.load_datasets()


"""Model parameters"""
u.printf ("Creating model parameters")
params = hp.ModelParameters(data_set.dp.max_classes)
params.training_batch_size=training_batch_size
params.learning_rate=learning_rate
params.dropout=dropout
params.epochs=epochs
params.convolutional_layer_count = convolutional_layer_count
params.check_batch_size(len(train_labels),params.training_batch_size)
params.check_batch_size(len(test_labels),params.testing_batch_size)

"""Memory parameters"""
u.printf ("Creating memory parameters")
mem_params = hp.MemoryParameters(params.epochs)


"""Session Variables"""
u.printf ("Initializing Neural Network...")
x = tf.placeholder('float32', [None, (params.image_size * params.image_size)])
y = tf.placeholder('int32')
keep_prob = tf.placeholder(tf.float32)

"""Neural Network Variables"""
cnn = cn.CNN(x, keep_prob, params.convolutional_layer_count, params.image_size, params.number_of_classes, neuron_multiplier)
## cnn = cn.CNN(x, 0.8, 2, 100, 10, 40)
prediction = cnn.return_network()

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer(params.learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

"""Examination code"""
prediction_examination = correct_prediction
prediction_value = tf.argmax(prediction, 1)
cost_examination = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y)


"""Train the neural network"""
def train_network(sess):
    sess.run(tf.global_variables_initializer())
    #train_image_subset = int(mem_params.mem_percentage * len(train_labels))
    #validation_image_subset = int(mem_params.mem_percentage * len(validate_labels))

    #total_iteration = 0
    #while total_iteration < mem_params.mem_segments:
    mem_iteration = 1
        #u.printf("Loading true data for iteration {} of {}...".format((total_iteration+1),int(mem_params.mem_segments)))
        #train_labels_real = data_set.load_label_data(train_labels, int(total_iteration*train_image_subset),
        #                                             train_image_subset)
        #train_data_real = data_set.load_image_data(train_file_names, int(total_iteration * train_image_subset),
        #                                           train_image_subset)
        #validation_labels_real = data_set.load_label_data(validate_labels, int(total_iteration*validation_image_subset),
        #                                                  validation_image_subset)
        #validation_data_real = data_set.load_image_data(validate_file_names, int(total_iteration*validation_image_subset),
         #                                               validation_image_subset)
    u.printf ("Beginning model training...")
    while mem_iteration <= params.epochs:
        mini_batch_x, mini_batch_y = data_set.next_batch(train_labels, train_data, params.training_batch_size)
        mini_batch_x = data_set.load_image_data(mini_batch_x,0,params.training_batch_size)
        mini_batch_y = data_set.load_label_data(mini_batch_y,0,params.training_batch_size, params.number_of_classes)

        sess.run(optimizer, feed_dict={x: mini_batch_x, y: mini_batch_y, keep_prob: params.dropout})

        #true_iteration = mem_iteration + (total_iteration * mem_params.mem_epochs)
        true_iteration=mem_iteration
        if mem_iteration % params.training_display == 0:
                # training_accuracy, training_loss, predictions, costs = sess.run([accuracy, cost, prediction_examination, cost_examination],
                # feed_dict={x: mini_batch_x, y: mini_batch_y, keep_prob: params.dropout})
            training_accuracy, training_loss = sess.run(
                    [accuracy, cost], feed_dict={x: mini_batch_x, y: mini_batch_y, keep_prob: params.dropout})
            u.printf ("Iteration: {}, Training Accuracy: {} Training Loss: {}".format(true_iteration, training_accuracy, training_loss))
            training_losses.append((training_loss,true_iteration))
            training_accuracies.append((training_accuracy,true_iteration))

        if mem_iteration % params.cross_validate == 0:
            validation_mini_batch_x, validation_mini_batch_y = data_set.next_batch(validation_labels, validation_data,
                                                                                       params.training_batch_size)

            validation_mini_batch_x = data_set.load_image_data(validation_mini_batch_x, 0, params.training_batch_size)
            validation_mini_batch_y = data_set.load_label_data(validation_mini_batch_y, 0, params.training_batch_size, params.number_of_classes)

            validation_accuracy, validation_loss = sess.run([accuracy, cost],
                                                     feed_dict={x: validation_mini_batch_x, y: validation_mini_batch_y,
                                                                keep_prob: params.dropout})
            u.printf ("Iteration: {}, Validation Accuracy: {}".format(true_iteration, validation_accuracy))
            validation_losses.append((validation_loss,true_iteration))
            validation_accuracies.append((validation_accuracy,true_iteration))

        mem_iteration += 1

        #total_iteration +=1


def test_network(sess):
    u.printf ("Testing network...")
    start_index = 0
    total_elements = len(test_labels)
    accuracy_values = []
    test_data_real = data_set.load_image_data(test_data, 0, total_elements)
    test_labels_real = data_set.load_label_data(test_labels, 0 ,total_elements, params.number_of_classes)
    while start_index < total_elements:
        end_index = start_index + params.testing_batch_size
        if end_index > total_elements:
            end_index = total_elements

        test_accuracy = sess.run(accuracy,
                                 feed_dict={x: test_data_real[start_index:end_index], y: test_labels_real[start_index:end_index],
                                            keep_prob: params.dropout})

        accuracy_values.append(test_accuracy)
        start_index += params.testing_batch_size

        if start_index % params.testing_display == 0:
            u.printf ('Tested {} data items'.format(end_index))

    u.printf ("Test Accuracy: {}".format(sum(accuracy_values) / float(total_elements/params.testing_batch_size)))


def generate_visuals():
    u.printf ("Generating visuals...")
    with tf.variable_scope('', reuse=True) as scope_conv:
        # test=tf.get_collection(tf.GraphKeys.VARIABLES)
        weights = tf.get_variable('dense/kernel')

    test_labels_real = data_set.load_label_data(test_labels, 0, params.visualize_images)
    test_data_real= data_set.load_image_data(test_data, 0, params.visualize_images)

    classmap = vs.get_map(0, cnn.return_final_convolutional_layer(), params.image_size, weights)
    vs.visualize_map(sess, classmap, cnn.return_final_convolutional_layer(), test_data_real, test_labels_real,
                     params.visualize_images, x, y, prediction, cnn.image_size, data_set.dp)

    vs.plot_model_training(training_losses, validation_losses , "Loss", data_set.dp)
    vs.plot_model_training(training_accuracies, validation_accuracies ,"Accuracy", data_set.dp)


if __name__ == '__main__':
    u.printf("Executing version {} of the code".format(version))
    u.printf ("Beginning model execution with the following settings\n" \
          "Convolutional-Layers:{} Epoch-Count:{} Image-Size:{} Batch-Size:{} Learning-Rate:{} Drop-Out:{} Display-Interval:{} Neurons:{}" \
          .format(params.convolutional_layer_count, params.epochs, params.image_size, params.training_batch_size,
                  params.learning_rate, params.dropout, params.training_display, neuron_multiplier))

    with tf.Session() as sess:
        train_network(sess)
        test_network(sess)
        generate_visuals()

    u.printf ("Finished model execution at {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
