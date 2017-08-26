import tensorflow as tf

class CNN:
    convolutional_layers = []
    #Kernals
    convolutional_kernel = 0
    pooling_kernel = 2
    #Size determinators
    layer_filter_increment=16
    neuron_multiplier=0
    #strides
    pool_stride = 2

    def __init__(self, inData, keep_rate, convolutional_layer_count, image_size, number_of_classes, neuron_multiplier, convolutional_filter):
        # Scalar variables
        self.image_size = image_size
        self.number_of_classes = number_of_classes
        self.neuron_multiplier = neuron_multiplier
        self.convolutional_kernel = convolutional_filter
        #Shape data
        self.data = inData
        self.shape_data(self.data)
        #Build convolutional layers
        previous_convolutional_layer=self.data
        for i in range(convolutional_layer_count):
            current_convolutional_layer = self.create_convolutional_layer(input_layer=previous_convolutional_layer,
                                                                    filters=(self.layer_filter_increment*(i+1)),
                                                                    convolutional_kernel=self.convolutional_kernel,
                                                                    pool_kernel=self.pooling_kernel,
                                                                    stride=self.pool_stride)
            previous_convolutional_layer = current_convolutional_layer
            self.convolutional_layers.append(current_convolutional_layer)
        #Terminate network
        self.neuron_count = (self.neuron_multiplier * (self.return_final_convolutional_layer().get_shape()[3]).value)
        self.terminate_network(keep_rate)

    def terminate_network(self,keep_rate):
        # Create fully connected layer
        self.full_network = self.create_fully_connected(self.return_final_convolutional_layer(), keep_rate)

    def shape_data(self, data):
        self.data = tf.reshape(data, shape=[-1, self.image_size, self.image_size, 1])

    def conv2d(self, input_layer, filters, kernel):
        return tf.layers.conv2d(inputs=input_layer, filters=filters, kernel_size=[kernel, kernel],  padding="same", activation=tf.nn.relu)

    def maxpool2d(self, input_layer, pool_size, stride):
        return tf.layers.max_pooling2d(inputs=input_layer, pool_size=[pool_size, pool_size], strides=stride)

    def create_convolutional_layer(self, input_layer, filters, convolutional_kernel, pool_kernel, stride):
        convolutional_layer = self.conv2d(input_layer=input_layer, filters=filters, kernel=convolutional_kernel)
        convolutional_layer = self.maxpool2d(input_layer=convolutional_layer, pool_size=pool_kernel, stride=stride)
        return convolutional_layer

    def create_fully_connected(self, layer, keep_rate):
        shape = layer.get_shape()
        flat_layer = tf.reshape(layer, [-1, int(shape[1]) * int(shape[2]) * int(shape[3])])
        dense = tf.layers.dense(inputs=flat_layer, units=self.neuron_count, activation=tf.nn.relu)
        dropout_layer = tf.layers.dropout(inputs=dense, rate=keep_rate)
        return tf.layers.dense(inputs=dropout_layer, units=self.number_of_classes)

    def return_final_convolutional_layer(self):
        return self.convolutional_layers[len(self.convolutional_layers)-1]

    def return_network(self):
        return self.full_network
