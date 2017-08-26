import numpy as np
from glob import glob
import random as rd
from sklearn.utils import shuffle
import scipy.misc as sc
import parameters as params
import util as u

class DataSet:

    def __init__(self, location, output):
        self.dp = params.DataSetParameters(location, output)

    def load_datasets(self):
        u.printf ("Loading file names...")
        dataset_images, dataset_labels = self.read_image_label_names()
        total_data = len(dataset_labels)

        test_end_index=int(total_data*self.dp.test_size)
        validate_end_index=int(test_end_index+(total_data*self.dp.val_size))
        train_end_index=total_data

        u.printf ("Randomizing loaded data...")
        dataset_labels, dataset_images = shuffle(dataset_labels, dataset_images)

        return dataset_labels[0:test_end_index],dataset_images[0:test_end_index], \
               dataset_labels[test_end_index:validate_end_index], dataset_images[test_end_index:validate_end_index], \
               dataset_labels[validate_end_index:train_end_index],dataset_images[validate_end_index:train_end_index]

    def load_label_datafile(self, label_names, start_position, to_load):
        data_set_labels = []
        number_loaded = 0
        number_of_files = len(label_names)
        image_position = start_position
        while image_position < number_of_files and number_loaded < to_load:
            with open(label_names[image_position], 'r') as label_file:
                for line in label_file:
                    data = line.replace("[", "")
                    data = data.replace("]", "")
                    data = data.replace("\n", "")
                    array = np.fromstring(data, dtype=int, sep=' ')
                    data_set_labels.append(array)

            number_loaded += 1
            image_position += 1

        return np.array(data_set_labels)

    def read_image_label_names(self):
        dataset_images = []
        dataset_labels = []
        number_loaded = 0
        number_of_files = len(glob('{}/*.{}'.format(self.dp.dataset_location,self.dp.input_file_ext)))
        for i in range(number_of_files):
            number_loaded += 1
            file_name='{}/{}_{}.{}'.format(self.dp.dataset_location, self.dp.image_prefix, i + 1, self.dp.input_file_ext)
            dataset_images.append(file_name)
            file_name = '{}/{}_{}.{}'.format(self.dp.dataset_location, self.dp.image_prefix, i + 1,"txt")
            dataset_labels.append(file_name)
            if number_loaded >= self.dp.samples_to_load != 0:
                break
            if number_loaded % self.dp.display == 0 and self.dp.display != 0:
                u.printf ("Loaded {} file names".format(number_loaded))

        u.printf ("Total file names: {}".format(i+1))
        return np.array(dataset_images), np.array(dataset_labels)

    def load_image_data(self, file_names, start_position, to_load):
        data_set_images = []
        number_loaded = 0
        number_of_files = len(file_names)
        image_position = start_position
        while image_position < number_of_files and number_loaded < to_load:
            unflattened_image=sc.imread(file_names[image_position],flatten=True).astype(float)
            data_set_images.append(unflattened_image.flatten())

            number_loaded += 1
            image_position += 1
        return np.array(data_set_images)

    @staticmethod
    def next_batch(labels, images, batch_size):
        data_size=len(labels)
        if data_size > batch_size:
            low_range = 0
            high_range=data_size-batch_size
            start_pos = rd.randint(low_range, high_range)
            return images[start_pos:(start_pos + batch_size)], labels[start_pos:(start_pos + batch_size)]
        else:
            return images, labels
