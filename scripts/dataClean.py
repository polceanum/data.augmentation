# script: Data cleaner. Reads video files and generates pickles of cropped objects.
# author: Mihai Polceanu

import cv2
import numpy as np
import os
import sys
import pickle
import argparse

def processImage(img):
    # the following two lines define the range of colors that is THROWN AWAY
    # change these values if you use a different background in the videos

    lower_red = np.array([100,100,100])
    upper_red = np.array([240,240,240])

    # obtain mask given the range
    mask = cv2.inRange(img, lower_red, upper_red)

    # retain regions given by mask
    res = cv2.bitwise_and(img,img, mask= mask)

    # remove regions calculated above
    res = img-res

    # result is the image regions that are NOT in the defined interval
    return res, cv2.bitwise_not(mask)

def cropObject(img, mask):
    # given that colors have been well selected, search for margins

    top = -1
    for i in range(img.shape[0]):
        if np.sum(mask[i,:]) > 0:
            top = i
            break

    bottom = -1
    for i in range(img.shape[0]):
        if np.sum(mask[img.shape[0]-i-1,:]) > 0:
            bottom = img.shape[0]-i-1
            break

    left = -1
    for j in range(img.shape[1]):
        if np.sum(mask[:,j]) > 0:
            left = j
            break

    right = -1
    for j in range(img.shape[1]):
        if np.sum(mask[:,img.shape[1]-j-1]) > 0:
            right = img.shape[1]-j-1
            break

    # result is the cropped region, given the found margins
    return img[top:bottom, left:right, :], mask[top:bottom, left:right]


print('OpenCV version', cv2.__version__)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data cleaner.')
    parser.add_argument('--skip', metavar='N', type=int, default=1, help='keep one in every N frames, default 1')
    parser.add_argument('--maxCropSize', metavar='C', type=int, default=480, help='max crop size (width or height), default 480')
    parser.add_argument('--display', metavar='D', type=int, default=1, help='show cropped objects (0==False, 1==True), default 1')
    parser.add_argument('--verbose', metavar='V', type=int, default=1, help='output text to console (0==False, 1==True), default 1')
    args = parser.parse_args()
    
    # safety check to avoid replacing previous data !
    if not os.path.exists('../generated_crop_data/'):
        # generate folders
        os.makedirs('../generated_crop_data/')

        # load all files in the video folder
        for root, dirs, files in os.walk("../dataset_videos"):  
            for filename in files:
                if args.verbose:
                    print('Processing file', filename)

                # start video capture
                vidcap = cv2.VideoCapture("../dataset_videos/"+filename)

                # read first frame
                success,image = vidcap.read()

                # label is the first string before '_'
                label = filename.split("_")[0]

                if args.verbose:
                    print('Video data shape', image.shape)

                count = 0
                success = True

                cropList = []
                while success:
                    skip = args.skip

                    if count % skip == 0:
                        if args.verbose:
                            print('frame',count)

                        img, mask = processImage(image)

                        kernel1 = np.ones((4,4), np.uint8)
                        kernel2 = np.ones((7,7), np.uint8)

                        mask = cv2.erode(mask, kernel1, iterations=2)

                        mask = cv2.dilate(mask, kernel2, iterations=2)

                        crop, cmask = cropObject(img, mask)

                        maxSize = max(crop.shape[0], crop.shape[1])

                        #ratio = 480.0/maxSize # all crops have max size (width or height) of 480 px
                        ratio = float(args.maxCropSize)/maxSize # all crops have maxCropSize (width or height)

                        small_crop = cv2.resize(crop, (int(crop.shape[1]*ratio), int(crop.shape[0]*ratio)))
                        small_cmask = cv2.resize(cmask, (int(cmask.shape[1]*ratio), int(cmask.shape[0]*ratio)))

                        cropList.append(small_crop)

                        if args.display:
                            cv2.imshow("Cropped object", small_crop)
                            cv2.waitKey(1)

                    #read next frame
                    success,image = vidcap.read()
                    count += 1

                # save pickle with crops
                with open('../generated_crop_data/'+filename+'_cropped.pickle', 'wb') as handle:
                    pickle.dump(cropList, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('---------------------------------------------------------------')
        print('| SANITY CHECK: ../generated_crop_data folder already exists! |')
        print('|        Please move or remove it to generate new data.       |')
        print('---------------------------------------------------------------')