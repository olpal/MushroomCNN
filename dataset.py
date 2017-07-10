import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import random as rd
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import scipy.misc as sc
import parameters as params
import util as u

class DataSet:

    def __init__(self, location, output):
        self.dp = params.DataSetParameters(location, output)

    def load_datasets(self):
        u.printf ("Loading labels...")
        dataset_labels = self.read_label_names()
        u.printf ("Loading data files...")
        dataset_images = self.read_image_names()
        total_data = len(dataset_labels)

        test_end_index=int(total_data*self.dp.test_size)
        validate_end_index=int(test_end_index+(total_data*self.dp.val_size))
        train_end_index=total_data

        #u.printf ("Randomizing loaded data...")
        #dataset_labels, dataset_images = shuffle(dataset_labels, dataset_images)

        return dataset_labels[0:test_end_index],dataset_images[0:test_end_index], \
               dataset_labels[test_end_index:validate_end_index], dataset_images[test_end_index:validate_end_index], \
               dataset_labels[validate_end_index:train_end_index],dataset_images[validate_end_index:train_end_index]

    def read_label_names_full(self):
        dataset_labels = []
        number_loaded=0
        with open('{}/labels.txt'.format(self.dp.dataset_location), 'r') as label_file:
            all_lines = label_file.readlines()
            for line in all_lines:
                number_loaded += 1
                [_, label] = line.strip().split('\t')
                dataset_labels.append(int(label))
                if number_loaded >= self.dp.samples_to_load != 0:
                    break

        matrix = np.asmatrix(dataset_labels).transpose()
        ohe = OneHotEncoder(n_values=(matrix.max() + 1))
        one_hot_matrix = ohe.fit_transform(matrix)
        self.dp.max_classes = one_hot_matrix.get_shape()[1]

        return one_hot_matrix.toarray()

    def read_image_names_full(self):
        dataset_images = []
        number_loaded = 0
        number_of_files = len(glob('{}/*.{}'.format(self.dp.dataset_location,self.dp.input_file_ext)))
        for i in range(number_of_files):
            number_loaded += 1
            file_name='{}/{}_{}.{}'.format(self.dp.dataset_location, self.dp.image_prefix, i + 1, self.dp.input_file_ext)
            unflattened_image = sc.imread(file_name, flatten=True).astype(float)
            dataset_images.append(unflattened_image.flatten())
            if number_loaded >= self.dp.samples_to_load != 0:
                break
            if number_loaded % self.dp.display == 0 and self.dp.display != 0:
                u.printf ("Loaded {} file names".format(number_loaded))

        u.printf ("Total file names: {}".format(i+1))
        return np.array(dataset_images)

    def read_label_names(self):
        dataset_labels = []
        number_loaded=0
        with open('{}/labels.txt'.format(self.dp.dataset_location), 'r') as label_file:
            all_lines = label_file.readlines()
            for line in all_lines:
                number_loaded += 1
                [_, label] = line.strip().split('\t')
                dataset_labels.append(int(label))
                if number_loaded >= self.dp.samples_to_load != 0:
                    break

        self.dp.max_classes = (max(dataset_labels)+1)

        return np.array(dataset_labels)

    def load_label_data(self, label_names, start_position, to_load, classes):
        end_position = (start_position + to_load)
        if end_position > len(label_names):
            end_position = len(label_names)

        matrix = np.asmatrix(np.array(label_names[start_position:end_position])).transpose()
        ohe = OneHotEncoder(n_values=classes)
        one_hot_matrix = ohe.fit_transform(matrix)
        self.max_classes = one_hot_matrix.get_shape()[1]

        return one_hot_matrix.toarray()

    def read_image_names(self):
        dataset_images = []
        number_loaded = 0
        number_of_files = len(glob('{}/*.{}'.format(self.dp.dataset_location,self.dp.input_file_ext)))
        for i in range(number_of_files):
            number_loaded += 1
            file_name='{}/{}_{}.{}'.format(self.dp.dataset_location, self.dp.image_prefix, i + 1, self.dp.input_file_ext)
            dataset_images.append(file_name)
            if number_loaded >= self.dp.samples_to_load != 0:
                break
            if number_loaded % self.dp.display == 0 and self.dp.display != 0:
                u.printf ("Loaded {} file names".format(number_loaded))

        u.printf ("Total file names: {}".format(i+1))
        return np.array(dataset_images)

    def load_image_data(self, file_names, start_position, to_load):
        data_set_images = []
        number_loaded = 0
        number_of_files = len(file_names)
        image_position = start_position
        while image_position < number_of_files and number_loaded < to_load:
            unflattened_image=sc.imread(file_names[image_position],flatten=True).astype(float)
            data_set_images.append(unflattened_image.flatten())

            #if number_loaded % self.dp.display == 0 and self.dp.display != 0:
            #    u.printf ("Loaded {} data items".format(number_loaded))

            number_loaded += 1
            image_position += 1

        #u.printf ("Total loaded data items: {}".format(number_loaded))
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
