import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import glob
import os
from skimage import draw
import numpy as np
from scipy import ndimage
import sys

xsize=256
ysize=256
xoverlap=80
yoverlap=80
image_count=1
imageCell=1
inputDirectory=""
outputDirectory=""
file=""
threshold=0
mode=1


def plot_image_rotate(slices, slices_rotated, in_image, filename):

    fileappend=filename.replace(".bmp","")
    csvarray = np.zeros((in_image.shape[0], in_image.shape[1]), dtype=int)
    for val in csvValues:
        if float(val[2]) < threshold:
            continue
        rr, cc = draw.circle(float(val[1]), float(val[0]), radius=float(val[2]), shape=csvarray.shape)
        csvarray[rr, cc] = 1

    np.set_printoptions(threshold=np.inf)
    global imageCell
    position=1
    to_process = in_image
    csv_data = csvarray
    while position <= 4:
        imageSlices = slices
        if position % 2 ==0:
            imageSlices = slices_rotated

        for imageSlice in imageSlices:
            saveFileName = ("{}_img_{}".format(fileappend,imageCell))
            imageCell += 1

            slicedImage = to_process[imageSlice[0]:imageSlice[1], imageSlice[2]:imageSlice[3]]
            slicedCsv = csv_data[imageSlice[0]:imageSlice[1], imageSlice[2]:imageSlice[3]]

            plt.imshow(slicedImage,cmap="gray")
            plt.imsave('{}images/{}.png'.format(outputDirectory,saveFileName), slicedImage, cmap='gray')

            labelfile = open('{}images/{}.txt'.format(outputDirectory, saveFileName), 'w')
            reshaped = np.reshape(slicedCsv, (65536,))
            labelfile.write("{}\n".format((np.array_str(reshaped, max_line_width=1000000))))
            labelfile.close()

            plt.clf()

        to_process = to_process.swapaxes(-2, -1)[..., ::-1]
        csv_data = csv_data.swapaxes(-2, -1)[..., ::-1]
        position+=1

def read_csv(filename):
    try:
        with open('{}csv/{}'.format(inputDirectory,filename.replace(".bmp",".csv")), 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                csvValues.append(row)
    except:
        print "No csv value found for file {}".format(fileName)

def slice_image(inimage):
    slices = []
    shape = inimage.shape
    yStartPos=0
    finalY=False
    while True:
        finalX=False
        xStartPos = 0
        while True:
            slices.append((yStartPos, (yStartPos + ysize), xStartPos, (xStartPos + xsize)))
            xStartPos = (xStartPos + (xsize-xoverlap))
            if (xStartPos+xsize) > shape[1] and finalX:
                break
            elif (xStartPos+xsize) > shape[1]:
                xStartPos = (shape[1] - xsize)
                finalX=True
        yStartPos = (yStartPos+(ysize - yoverlap))
        if (yStartPos+ysize) > shape[0] and finalY:
            break
        elif (yStartPos+ysize) > shape[0]:
            yStartPos = (shape[0] - ysize)
            finalY = True

    return slices

def rename_match():
    image_files = glob.glob('{}{}/*.png'.format(outputDirectory, 'images'))
    index=1
    for i in xrange(len(image_files)):
        label_name = image_files[i].replace("png","txt")
        image_file_name="{}{}/img_{}.{}".format(outputDirectory,"data",index,"png")
        label_file_name="{}{}/img_{}.{}".format(outputDirectory,"data",index,"txt")
        os.rename(image_files[i],image_file_name)
        os.rename(label_name, label_file_name)
        index+=1

if __name__ == '__main__':

    if len(sys.argv) > 1:
        print ("Arguments found, loading...")
        mode = int(sys.argv[1])
        file = sys.argv[2]

    if mode == 1:
        rename_match()
    elif mode == 2:
        fileName = os.path.basename(file)
        image = plt.imread(file)
        csvValues = []
        dataArrays = []
        read_csv(fileName)
        if not csvValues:
            exit()
        imageSlices = slice_image(image)
        imageSliceRotated = slice_image(ndimage.rotate(image,90))
        plot_image_rotate(imageSlices, imageSliceRotated, image, fileName)



