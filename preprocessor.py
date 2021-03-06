import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import glob
import os
from skimage import draw
import numpy as np
from scipy import ndimage
from scipy import misc
import sys

xsize=256
ysize=256
xoverlap=80
yoverlap=80
image=""
image_count=1
imageCell=1
inputDirectory=""
outputDirectory=""
resultDirectoy=""
image_output_count=1
overlay_images=False
write_original=False
threshold=40
mode=3


def draw_circles(in_image):
    csvarray = np.zeros((in_image.shape[0], in_image.shape[1]), dtype=int)
    for val in csv_values:
        if float(val[2]) < threshold:
            continue
        rr, cc = draw.circle(float(val[1]), float(val[0]), radius=float(val[2]), shape=csvarray.shape)
        csvarray[rr, cc] = 1

    return csvarray

def plot_image_rotate(slices, slices_rotated, in_image):
    np.set_printoptions(threshold=np.inf)
    global imageCell
    #Rotation position
    position=1
    csv_data = draw_circles(in_image)
    while position <= 4:
        imageSlices = slices
        #If it has been rotated 90 or 270 degrees
        if position % 2 ==0:
            imageSlices = slices_rotated

        if write_original == True:
            plt.imshow(in_image, cmap="gray")
            plt.imsave('{}images/original_{}.png'.format(outputDirectory,imageCell), in_image, cmap='gray')

        for imageSlice in imageSlices:
            saveFileName = ("img_{}".format(imageCell))
            imageCell += 1

            slicedImage = in_image[imageSlice[0]:imageSlice[1], imageSlice[2]:imageSlice[3]]
            slicedCsv = csv_data[imageSlice[0]:imageSlice[1], imageSlice[2]:imageSlice[3]]

            plt.imshow(slicedImage,cmap="gray")
            plt.imsave('{}images/{}.png'.format(outputDirectory,saveFileName), slicedImage, cmap='gray')

            labelfile = open('{}images/{}.txt'.format(outputDirectory, saveFileName), 'w')
            reshaped = np.reshape(slicedCsv, (65536,))
            labelfile.write("{}\n".format((np.array_str(reshaped, max_line_width=1000000))))
            labelfile.close()

            if overlay_images:
                slicedImage = np.reshape(slicedImage, (65536,))
                for i in xrange(65536):
                    if reshaped[i] == 1:
                        slicedImage[i] = 1
                plt.imsave('{}images/{}_overlay.png'.format(outputDirectory,saveFileName), np.reshape(slicedImage,(256,256)), cmap='gray')

            plt.clf()
       #Rotate the matrices by 90 degrees
        to_process = to_process.swapaxes(-2, -1)[..., ::-1]
        csv_data = csv_data.swapaxes(-2, -1)[..., ::-1]
        position+=1

def read_csv(filename):
    try:
        with open('{}csv/{}'.format(inputDirectory,filename.replace(".bmp",".csv")), 'rb') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                csv_values.append(row)
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

def rename_files():
    for file in glob.glob('{}{}/*_.bmp'.format(inputDirectory,'images')):
        os.rename(file,file.replace("_.bmp",".bmp"))

if __name__ == '__main__':

    files = glob.glob('{}{}/*.bmp'.format(inputDirectory,'images'))
    print ("Found {} files".format(len(files)))
    file_count=1
    for file in files:
        fileName = os.path.basename(file)
        image = plt.imread(file)
        csv_values = []
        dataArrays = []
        read_csv(fileName)
        if not csv_values:
            continue
        imageSlices = slice_image(image)
        imageSliceRotated = slice_image(ndimage.rotate(image,90))
        plot_image_rotate(imageSlices, imageSliceRotated, image)
        if file_count % 10 == 0:
            print ("Processed {} files".format(file_count))
        file_count+=1
