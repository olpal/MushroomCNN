#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import cnn as cn
import parameters as hp
import sys
import util as u
import csv
import numpy as np

version=2.0


class Model():

    def __init__(self, model_directory,input_image, output_directory):
        u.printf ("Creating model parameters")
        self.train_params = hp.ModelParameters()
        self.exec_params = hp.ExecutionParameters()

        u.printf ("Initializing Neural Network parameters...")
        self.x = tf.placeholder('float32', [None, (self.train_params.image_size * self.train_params.image_size)])
        self.keep_prob = tf.placeholder(tf.float32)

        """Assign passed in variables if they exist"""
        self.exec_params.model_directory = model_directory
        self.exec_params.input_image = input_image
        self.exec_params.output_directory = output_directory

    def load_model(self):
        parameters=[]
        self.exec_params.variable_full_path='{}/{}'.format(self.exec_params.model_directory,self.train_params.saved_variable_file_name)
        self.exec_params.model_full_path = '{}/{}'.format(self.exec_params.model_directory, self.train_params.saved_model_file_name)
        try:
            with open(self.exec_params.variable_full_path, 'rb') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                    parameters.append(row)
                self.train_params.number_of_classes = int(parameters[0][0])
                self.train_params.neuron_multiplier = int(parameters[0][1])
                self.train_params.convolutional_filter = int(parameters[0][2])
                self.train_params.convolutional_layer_count = int(parameters[0][3])
                return True
        except:
            print "No file found at location {}".format(self.exec_params.variable_full_path)
            return False

    def create_network(self):
        """Neural Network Variables"""
        cnn = cn.CNN(self.x, self.keep_prob, self.train_params.convolutional_layer_count, self.train_params.image_size, self.train_params.number_of_classes,
                     self.train_params.neuron_multiplier, self.train_params.convolutional_filter)
        prediction = cnn.return_network()
        return tf.round(tf.nn.sigmoid(prediction))

    def segment_image(self,image):
        slices = []
        shape = image.shape
        yStartPos = 0
        finalY = False
        while True:
            finalX = False
            xStartPos = 0
            while True:
                slices.append((yStartPos, (yStartPos + self.exec_params.y_size), xStartPos, (xStartPos + self.exec_params.x_size)))
                xStartPos = (xStartPos + (self.exec_params.x_size - self.exec_params.x_overlap))
                if (xStartPos + self.exec_params.x_size) > shape[1] and finalX:
                    break
                elif (xStartPos + self.exec_params.x_size) > shape[1]:
                    xStartPos = (shape[1] - self.exec_params.x_size)
                    finalX = True
            yStartPos = (yStartPos + (self.exec_params.y_size - self.exec_params.y_overlap))
            if (yStartPos + self.exec_params.y_size) > shape[0] and finalY:
                break
            elif (yStartPos + self.exec_params.y_size) > shape[0]:
                yStartPos = (shape[0] - self.exec_params.y_size)
                finalY = True

        return slices


    def stitch_image(self, slices, predictions, shape):
        predicted_image = np.zeros((shape[0],shape[1]),dtype=int)
        image_position=0
        for slice in slices:
            predicted_image[slice[0]:slice[1], slice[2]:slice[3]] = np.reshape(predictions[image_position],(self.train_params.image_size,self.train_params.image_size))
            image_position+=1
        return predicted_image


    def predict_images(self, sess, network, image, slices):
        predicted_images=[]
        for slice in slices:
            to_process=[]
            to_process_image = image[slice[0]:slice[1], slice[2]:slice[3]]
            to_process.append(np.reshape(to_process_image, (self.train_params.image_size*self.train_params.image_size,)))
            predicted_images.append(sess.run([network], feed_dict={self.x: to_process, self.keep_prob: 1.}))
        return predicted_images

    def export_data(self, image_data, shape):
        np.set_printoptions(threshold=np.inf)
        image_file = open('{}/predicted_data.txt'.format(self.exec_params.output_directory), 'w')

        reshaped_prediction = np.reshape(image_data, (shape[0] * shape[1],))
        reshaped_prediction = reshaped_prediction.astype(int)

        image_file.write("{}\n".format((np.array_str(reshaped_prediction, max_line_width=1000000))))
        image_file.close()

    def export_image(self, image_data):
        plt.imshow(image_data, cmap="gray")
        plt.imsave('{}/predicted_image.png'.format(self.exec_params.output_directory), image_data, cmap="gray", dpi=250)

    def execute(self):
        u.printf("Executing version {} of the code".format(version))
        if not self.load_model():
            exit()
        network = self.create_network()
        image = plt.imread(self.exec_params.input_image)
        slices = self.segment_image(image)

        u.printf("Predicting image...")
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.exec_params.model_full_path)
            predicted_images=self.predict_images(sess, network, image, slices)

        u.printf("Stitching image")
        full_predicted_image = self.stitch_image(slices,predicted_images,image.shape)
        if self.exec_params.export_data:
            u.printf("Exporting data")
            self.export_data(full_predicted_image, image.shape)
        if self.exec_params.export_image:
            u.printf("Exporting image")
            self.export_image(full_predicted_image)