from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from scipy import misc
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import util as u
import sys
import parameters as hp
import numpy as np
import csv

params = hp.ImageProcessingParameters()

def load_parameters():
    """Assign passed in variables if they exist"""
    if len(sys.argv) > 1:
        u.printf ("Arguments found, loading basic params from command line...")
        params.input_file = sys.argv[1]
        params.output_dir = sys.argv[2]
        params.mode = int(sys.argv[3])
        params.image_prefix = sys.argv[4]
        params.image_postfix = int(sys.argv[5])
        if len(sys.argv) <= 5:
            return
        u.printf("Additional Arguments found, loading algorithm params from command line...")
        params.min_sigma = int(sys.argv[6])
        params.max_sigma = int(sys.argv[7])
        params.num_sigma = int(sys.argv[8])
        params.threshold = float(sys.argv[9])
        params.overlap = float(sys.argv[10])
        params.log_Scale = bool(sys.argv[11])
        params.sigma_ratio = float(sys.argv[12])



def set_params_log():
    u.printf("Loading params for LoG processing...")
    params.min_sigma = 20
    params.max_sigma = 40
    params.num_sigma = 5
    params.threshold = 0.2
    params.overlap = 0.7
    params.log_Scale = False


def set_params_dog():
    u.printf("Loading params for DoG processing...")
    params.min_sigma = 20
    params.max_sigma = 50
    params.threshold = 2.0
    params.overlap = 0.5
    params.sigma_ratio = 1.6


def set_params_doh():
    u.printf("Loading params for DoH processing...")
    params.min_sigma = 15
    params.max_sigma = 60
    params.num_sigma = 15
    params.threshold = 0.01
    params.overlap = 0.5
    params.log_Scale = True


def load_data():
    u.printf("Loading image...")
    global image
    image = misc.imread(params.input_file, flatten=True).astype(np.uint8)


def run_log():
    u.printf("Running Laplacian of Gaussian")
    data_matrix = blob_log(image, min_sigma=params.min_sigma, max_sigma=params.max_sigma, num_sigma=params.num_sigma, \
                           overlap=params.overlap, log_scale=params.log_Scale, threshold=params.threshold)
    data_matrix[:, 2] = data_matrix[:, 2] * sqrt(2)
    return data_matrix


def run_dog():
    u.printf("Running Difference of Gaussian")
    data_matrix = blob_dog(image, min_sigma=params.min_sigma, max_sigma=params.max_sigma, sigma_ratio=params.sigma_ratio,
                           overlap=params.overlap, threshold=params.threshold)
    data_matrix[:, 2] = data_matrix[:, 2] * sqrt(2)
    return data_matrix


def run_doh():
    u.printf("Running Determinant of Hessian")
    return blob_doh(image, min_sigma=params.min_sigma, max_sigma=params.max_sigma, num_sigma=params.num_sigma,
                    overlap=params.overlap, log_scale=params.log_Scale, threshold=params.threshold)


def display_data(in_image, circles, output_path):
    plt.clf()
    plt.imshow(in_image, interpolation='nearest', cmap="gray")
    fig = plt.gcf()
    axix = fig.gca()
    for circle in circles:
        y, x, r = circle
        circle = plt.Circle((x, y), r, color="red", linewidth=1, fill=False)
        axix.add_artist(circle)
    if params.display:
        plt.show()
    else:
        fig.savefig(output_path, dpi=250)

def export_data(circles, output_file):
    try:
        with open(output_file, 'w') as csv_file:
            writer = csv.writer(csv_file,delimiter=",")
            for circle in circles:
                y, x, r = circle
                writer.writerow([x,y,r])
    except:
        u.printf("Unable to write data to file: {}".format(output_file))



def statistics():
    u.printf("Running processing with parameters: Min Sigma:{} Max Sigma:{} Num Sigma:{} Overlap:{} Sigma Ratio:{}"
             " Threshold:{} Log Scale:{} Image Postfix:{}".format(params.min_sigma, params.max_sigma, params.num_sigma,
                                                                  params.overlap, params.sigma_ratio, params.threshold,
                                                                  params.log_Scale, params.image_postfix))

if __name__ == '__main__':
    load_parameters()
    load_data()
    execution_type=""
    if params.mode==1:
        if len(sys.argv) <= 1:
            set_params_log()
        statistics()
        data = run_log()
        execution_type="log"
    elif params.mode==2:
        if len(sys.argv) <= 1:
            set_params_dog()
        statistics()
        data = run_dog()
        execution_type="dog"
    elif params.mode==3:
        if len(sys.argv) <= 1:
            set_params_doh()
        statistics()
        data = run_doh()
        execution_type="doh"

    u.printf("Generating blob image")
    display_data(image, data, ("{}/{}_{}_{}.pdf".format(params.output_dir, execution_type,
                                                        params.image_prefix ,params.image_postfix)))
    if params.export_overlay:
        u.printf("Creating overlay image")
        overlay_image = misc.imread(params.overlay_file)
        display_data(overlay_image, data, ("{}/{}_{}_overlay_{}.pdf".format(params.output_dir, execution_type,
                                                                            params.image_prefix, params.image_postfix)))
    if params.export_data:
        u.printf("Exporting CSV data")
        export_data(data, ("{}/{}_{}_data_{}.csv".format(params.output_dir, execution_type,
                                                                            params.image_prefix, params.image_postfix)))

