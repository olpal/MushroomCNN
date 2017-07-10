import util as u
import os

class MemoryParameters():
    mem_percentage=1
    mem_segments=0
    mem_epochs=0

    def __init__(self, total_epochs):
        #inttialize memory segments variable
        if 1 % self.mem_percentage == 0:
            self.mem_segments = 1/self.mem_percentage
        else:
            self.mem_segments = (1 / self.mem_percentage) + 1
        #Initialize memory epocs
        self.mem_epochs = total_epochs * self.mem_percentage

class DataSetParameters():
    image_prefix = "img"
    input_file_ext = "png"
    output_file_ext = "png"
    output_graph_ext = "pdf"
    train_size = 0.7
    test_size = 0.15
    val_size = 0.15
    samples_to_load = 0
    display = 5000
    max_classes = 0

    def __init__(self, location, output):
        self.dataset_location = location
        self.output_dir = output
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)


class ModelParameters():
    """Hyper-parameters"""
    training_batch_size = 0
    learning_rate=0
    dropout = 0
    epochs=0
    convolutional_layer_count = 0
    """Display variables"""
    training_display=100
    cross_validate=100
    testing_display = 1000
    """Fixed Variables"""
    image_size=256
    number_of_classes=0
    visualize_images = 10
    testing_batch_size = 100


    def __init__(self, max_classes):
        self.check_convolutional_layers()
        self.number_of_classes = max_classes

    def check_convolutional_layers(self):
        u.printf ("Validating convolutional layers...")
        previous_size = self.image_size
        for i in range(self.convolutional_layer_count):
            current_size = previous_size / 2
            if current_size != 1:
                previous_size = current_size
                continue
            u.printf ("Maximum Convolutional Layer value reached\nSetting Convolutional Layer Count value to maximum: {}".format(i))
            self.convolutional_layer_count = i
            break

    def check_batch_size(self, data_set_size, batch_size):
        u.printf ("Validating batch size of {}...".format(batch_size))
        original_batch_size=batch_size
        if batch_size > data_set_size:
            while batch_size > data_set_size:
                batch_size /= 2
            u.printf ("Batch size {} is greater than data set size {}\nBatch size has been reduced to {}".format(original_batch_size,data_set_size,batch_size))