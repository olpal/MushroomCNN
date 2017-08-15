import util as u
import os
import datetime

class ImageProcessingParameters():
    # DOG,LOG,DOH - The minimum standard deviation for Gaussian Kernel. Keep this low to detect smaller blobs
    min_sigma = 0
    # DOG,LOG,DOH - The maximum standard deviation for Gaussian Kernel. Keep this high to detect larger blobs
    max_sigma = 0
    # LOG,DOH - The number of intermediate values of standard deviations to consider between min_sigma and max_sigma
    num_sigma = 0
    # DOG,LOG,DOH - The absolute lower bound for scale space maxima. Local maxima smaller than thresh are ignored.
    # Reduce this to detect blobs with less intensities.
    threshold = 0.0
    # DOG,LOG,DOH - A value between 0 and 1. If the area of two blobs overlaps by a fraction greater than threshold,
    # the smaller blob is eliminated"""
    overlap = 0.0
    # LOG,DOH - If set intermediate values of standard deviations are interpolated using a logarithmic scale to
    # the base 10. If not, linear interpolation is used
    log_Scale = False
    # DOG - The ratio between the standard deviation of Gaussian Kernels used for computing the Difference of Gaussians
    sigma_ratio = 0.0
    image_postfix = 1
    image_prefix = ""
    display = False
    export_overlay = True
    export_data = True
    mode = 1
    input_file = "/Users/aolpin/Documents/School/thesis/results/2017-08-13--13-03-00-916973452/predicted_image.png"
    overlay_file = "/Users/aolpin/Documents/School/thesis/data/images/1_2016-11-15-11_00.bmp"
    output_dir = "/Users/aolpin/Documents/School/thesis/results/2017-08-13--13-03-00-916973452"

class DataSetParameters():
    """File extensions"""
    image_prefix = "img"
    input_file_ext = "png"
    output_file_ext = "png"
    output_graph_ext = "pdf"
    """Variables"""
    train_size = 0.7
    test_size = 0.15
    val_size = 0.15
    samples_to_load = 0
    display = 5000
    max_classes = 0

    def __init__(self, location, output):
        self.dataset_location = location
        self.output_dir=output
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

class ExecutionParameters():
    model_directory = "/Users/aolpin/Documents/School/thesis/results/2017-08-13--13-03-00-916973452"
    variable_full_path=""
    model_full_path=""
    input_image = "/Users/aolpin/Documents/School/thesis/dataset-sig/images/1_2016-11-15-11_00.bmp"
    x_size = 256
    y_size = 256
    x_overlap = 80
    y_overlap = 80
    export_data=False
    export_image=True


class ModelParameters():
    """File Paths"""
    dataset_location = "/Users/aolpin/Documents/School/thesis/data-sig/images"
    output_dir = "/Users/aolpin/Documents/School/thesis/results/{}".format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    saved_model_file_name = "model.ckpt"
    saved_variable_file_name = "variables.txt"
    saved_model_file_path = "/tmp"
    saved_variable_file_path = "/tmp"
    """Display variables"""
    training_display=100
    cross_validate=100
    testing_display = 1000
    """Fixed Variables"""
    image_size=256
    number_of_classes=image_size*image_size
    visualize_images = 30
    testing_batch_size = 100
    """Default model variables"""
    """These will be overridden with passed in variables"""
    training_batch_size = 100
    learning_rate = 0.001
    dropout = 0.8
    epochs = 1
    convolutional_layer_count = 3
    neuron_multiplier = 40
    convolutional_filter = 5
    save_model = True


    def __init__(self):
        self.check_convolutional_layers()

    def generate_model_paths(self):
        self.saved_model_file_path = "{}/{}".format(self.output_dir, self.saved_model_file_name)
        self.saved_variable_file_path = "{}/{}".format(self.output_dir, self.saved_variable_file_name)

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