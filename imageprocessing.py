from math import sqrt
from skimage.feature import blob_dog, blob_log, blob_doh
from scipy import misc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import util as u
import sys

min_sigma=1
max_sigma=15
num_sigma=0
threshold=0.01
overlap=0.1
log_Scale=False
sigma_ratio=1.1
image_postfix=1
display=False
mode=2

imput_file="/Users/aolpin/Documents/School/thesis/data/images/1_2016-11-15-11_00.bmp"
output_dir="/Users/aolpin/Documents/School/thesis/imageprocessing/"

"""Assign passed in variables if they exist"""
if len(sys.argv) > 1:
    u.printf ("Arguments found, loading...")
    imput_file = sys.argv[1]
    output_dir = sys.argv[2]
    mode = int(sys.argv[3])
    min_sigma = int(sys.argv[4])
    max_sigma = int(sys.argv[5])
    num_sigma = int(sys.argv[6])
    threshold = float(sys.argv[7])
    overlap = float(sys.argv[8])
    log_Scale = bool(sys.argv[9])
    sigma_ratio = float(sys.argv[10])
    image_postfix = int(sys.argv[11])


def load_data():
    global image
    image = misc.imread(imput_file)


def run_log():
    print("Running Laplacian of Gaussian")
    data_matrix = blob_log(image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, overlap=overlap, log_scale=log_Scale, threshold=threshold)
    data_matrix[:, 2] = data_matrix[:, 2] * sqrt(2)
    return data_matrix

def run_dog():
    print("Running Difference of Gaussian")
    data_matrix = blob_dog(image, min_sigma=min_sigma, max_sigma=max_sigma, sigma_ratio=sigma_ratio, overlap=overlap, threshold=threshold)
    data_matrix[:, 2] = data_matrix[:, 2] * sqrt(2)
    return data_matrix

def run_doh():
    print("Running Determinant of Hessian")
    return blob_doh(image, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, overlap=overlap, log_scale=log_Scale, threshold=threshold)


def display_data(in_image, output_path):
    print("Displaying data")
    plt.clf()
    plt.imshow(image, interpolation='nearest', cmap="gray")
    fig = plt.gcf()
    axix = fig.gca()
    for circle in in_image:
        y, x, r = circle
        circle = plt.Circle((x, y), r, color="red", linewidth=1, fill=False)
        axix.add_artist(circle)
    if display:
        plt.show()
    else:
        fig.savefig(output_path, dpi=250)

if __name__ == '__main__':
    u.printf("Running image processing with parameters: Min Sigma:{} Max Sigma:{} Num Sigma:{} Overlap:{} Sigma Ratio:{} "
             "Threshold:{} Log Scale:{} Image Postfix:{}".format(min_sigma,max_sigma,num_sigma,overlap,sigma_ratio, threshold, log_Scale, image_postfix))
    load_data()
    if mode==1:
        data = run_log()
        display_data(data,("{}log_{}.pdf".format(output_dir,image_postfix)))
    elif mode==2:
        data = run_dog()
        display_data(data, ("{}dog_{}.pdf".format(output_dir, image_postfix)))
    elif mode==3:
        data = run_doh()
        display_data(data, ("{}doh_{}.pdf".format(output_dir, image_postfix)))
