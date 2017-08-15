import datetime
import tensorflow as tf
import cnn as cn
import visualize as vs
import dataset as ds
import parameters as hp
import sys
import util as u
import datetime as dt
import csv

version = 4.0

u.printf ("Starting model execution at {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

"""Variables"""
u.printf ("Intializing list variables...")
training_losses = []
training_accuracies = []
validation_losses = []
validation_accuracies = []

u.printf ("Creating model parameters")
params = hp.ModelParameters()

"""Assign passed in variables if they exist"""
if len(sys.argv) > 1:
    u.printf ("Arguments found, loading...")
    params.dataset_location = sys.argv[1]
    params.output_dir = sys.argv[2]
    params.training_batch_size = int(sys.argv[3])
    params.learning_rate = float(sys.argv[4])
    params.dropout = float(sys.argv[5])
    params.epochs = int(sys.argv[6])
    params.convolutional_layer_count = int(sys.argv[7])
    params.neuron_multiplier = int(sys.argv[8])
    params.convolutional_filter = int(sys.argv[9])

"""Data sets"""
u.printf ("Building dataset...")
data_set = ds.DataSet(params.dataset_location, params.output_dir)
test_labels, test_data, validation_labels, validation_data, train_labels, train_data = data_set.load_datasets()
params.check_batch_size(len(train_labels),params.training_batch_size)
params.check_batch_size(len(test_labels),params.testing_batch_size)
params.generate_model_paths()

"""Session Variables"""
u.printf ("Initializing Neural Network...")
x = tf.placeholder('float32', [None, (params.image_size * params.image_size)])
y = tf.placeholder('float32')
keep_prob = tf.placeholder(tf.float32)

"""Neural Network Variables"""
cnn = cn.CNN(x, keep_prob, params.convolutional_layer_count, params.image_size, params.number_of_classes, params.neuron_multiplier, params.convolutional_filter)
prediction = cnn.return_network()

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer(params.learning_rate).minimize(cost)

"""Overall Accuracy """
correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(prediction)), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

"""All true Accuracy """
all_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
accuracy_all_true = tf.reduce_mean(all_true)


"""Examination code"""
prediction_truth = correct_prediction
prediction_true_values = tf.nn.sigmoid(prediction)
prediction_round_values = tf.round(tf.nn.sigmoid(prediction))
cost_examination = tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction, labels=y)



"""Train the neural network"""
def train_network(sess):
    sess.run(tf.global_variables_initializer())
    current_epoch = 1
    u.printf ("Beginning model training...")
    while current_epoch <= params.epochs:
        mini_batch_x, mini_batch_y = data_set.next_batch(train_labels, train_data, params.training_batch_size)
        mini_batch_x = data_set.load_image_data(mini_batch_x,0,params.training_batch_size)
        mini_batch_y = data_set.load_label_datafile(mini_batch_y,0,params.training_batch_size)

        sess.run(optimizer, feed_dict={x: mini_batch_x, y: mini_batch_y, keep_prob: params.dropout})


        if current_epoch % params.training_display == 0:
            #training_accuracy, training_loss, predictions, costs = sess.run([accuracy, cost, prediction_true_values, cost_examination], feed_dict={x: mini_batch_x, y: mini_batch_y, keep_prob: params.dropout})
            training_accuracy,training_accuracy2, training_loss = sess.run(
                    [accuracy, accuracy_all_true, cost], feed_dict={x: mini_batch_x, y: mini_batch_y, keep_prob: 1.0})
            u.printf ("Iteration: {}, Training Accuracy: {} - {} Training Loss: {}".format(current_epoch, training_accuracy, training_accuracy2, training_loss))
            training_losses.append((training_loss,current_epoch))
            training_accuracies.append((training_accuracy,current_epoch))

        if current_epoch % params.cross_validate == 0:
            validation_mini_batch_x, validation_mini_batch_y = data_set.next_batch(validation_labels, validation_data,
                                                                                       params.training_batch_size)

            validation_mini_batch_x = data_set.load_image_data(validation_mini_batch_x, 0, params.training_batch_size)
            validation_mini_batch_y = data_set.load_label_datafile(validation_mini_batch_y, 0, params.training_batch_size)

            validation_accuracy, validation_loss = sess.run([accuracy, cost],
                                                     feed_dict={x: validation_mini_batch_x, y: validation_mini_batch_y,
                                                                keep_prob: 1.0})
            u.printf ("Iteration: {}, Validation Accuracy: {}".format(current_epoch, validation_accuracy))
            validation_losses.append((validation_loss,current_epoch))
            validation_accuracies.append((validation_accuracy,current_epoch))

        current_epoch += 1


def test_network(sess):
    u.printf ("Testing network...")
    start_index = 0
    total_elements = len(test_labels)
    accuracy_values = []
    while start_index < total_elements:

        test_data_real = data_set.load_image_data(test_data, start_index, params.testing_batch_size)
        test_labels_real = data_set.load_label_datafile(test_labels, start_index, params.testing_batch_size)

        test_accuracy = sess.run(accuracy, feed_dict={x: test_data_real, y: test_labels_real, keep_prob: 1.0})

        accuracy_values.append(test_accuracy)

        if start_index % params.testing_display == 0:
            u.printf ('Tested {} data items'.format(start_index))

        start_index += params.testing_batch_size

    u.printf ("Test Accuracy: {}".format(sum(accuracy_values) / float(total_elements/params.testing_batch_size)))


def generate_visuals():
    u.printf ("Generating visuals...")

    test_labels_real = data_set.load_label_datafile(test_labels, 0, params.visualize_images)
    test_data_real= data_set.load_image_data(test_data, 0, params.visualize_images)

    vs.visualize_map(sess, test_data_real, test_labels_real, cnn.image_size, data_set.dp,
                     prediction_true_values, x, y, keep_prob, params)

    vs.visualize_data(sess, test_data_real, test_labels_real, prediction_round_values, x, y, keep_prob,
                      data_set.dp, params)

    vs.plot_model_training(training_losses, validation_losses , "Loss", data_set.dp)
    vs.plot_model_training(training_accuracies, validation_accuracies ,"Accuracy", data_set.dp)


def save_model(sess):
    model_saver = tf.train.Saver()
    model_saver.save(sess, params.saved_model_file_path)


def save_variables():
    try:
        with open(params.saved_variable_file_path, 'w') as csvfile:
            writer = csv.writer(csvfile,delimiter=",")
            writer.writerow([params.number_of_classes,params.neuron_multiplier,params.convolutional_filter,params.convolutional_layer_count])
    except:
        u.printf("Unable to write to variable file: {}".format(params.saved_variable_file_path))


if __name__ == '__main__':
    u.printf("Executing version {} of the code".format(version))
    u.printf ("Beginning model execution with the following settings\n" \
          "Convolutional-Layers:{} Epoch-Count:{} Image-Size:{} Batch-Size:{} Learning-Rate:{} Drop-Out:{} "
              "Display-Interval:{} Neuron Multiplier:{} Convolutional Filter:{}" \
          .format(params.convolutional_layer_count, params.epochs, params.image_size, params.training_batch_size,
                  params.learning_rate, params.dropout, params.training_display, params.neuron_multiplier, params.convolutional_filter))

    with tf.Session() as sess:
        train_network(sess)
        test_network(sess)
        generate_visuals()
        if params.save_model:
            save_model(sess)
            save_variables()


    u.printf ("Finished model execution at {}".format(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
