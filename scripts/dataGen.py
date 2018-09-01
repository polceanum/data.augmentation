# script: Data generator. Reads cropped objects pickles and background images and generates image datasets.
# author: Mihai Polceanu

import cv2
import numpy as np
import os
import sys
import pickle
import random
import imutils
import argparse

def rndint(l,h):
    return np.random.randint(l, h)

def resize(img):
    ratio = np.random.uniform(0.01, 0.5)
    noiseX = np.random.uniform(0.0, 0.1)
    noiseY = np.random.uniform(0.0, 0.1)
    result = cv2.resize(img, (int(img.shape[1]*(ratio+noiseX)), int(img.shape[0]*(ratio+noiseY))))
    return result

def rotate(img):
    angle = np.random.randint(0, 360)
    result = imutils.rotate_bound(img, angle)
    return result

def skew(img):
    rows,cols,ch = img.shape

    a = np.random.uniform(0, 10)
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[50+rndint(0,a),50+rndint(0,a)],[200+rndint(0,a),50+rndint(0,a)],[50+rndint(0,a),200+rndint(0,a)]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))

    return dst

def noise(img):
    amount = 20
    n = np.random.uniform(-amount, amount, img.shape)
    
    result = img.astype(np.float32)+n
    result[result<0] = 0
    result[result>255] = 255
    result = result.astype(np.uint8)
    return result

def color(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cropMask = np.sum(img.copy().astype(np.float32), axis=2).reshape(img.shape[0], img.shape[1], 1)
    cropMask[cropMask > 0] = 1

    #cv2.imshow("crop", cropMask)

    amount = 40
    
    #print(img.shape)
    n = np.random.uniform(-amount, amount, (1, 1, 1))
    #print(n.shape)
    
    result = hsv_img.astype(np.float32)
    result[:, :, 0:1] += n
    result[result[:,:,0:1]<0] += 179
    result[result[:,:,0:1]>179] -= 179
    result = result*cropMask
    
    result = result.astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    return result

def color2(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cropMask = np.sum(img.copy().astype(np.float32), axis=2).reshape(img.shape[0], img.shape[1], 1)
    cropMask[cropMask > 0] = 1

    #cv2.imshow("crop", cropMask)

    amount = 10
    amount2 = 20
    
    #print(img.shape)
    n = np.random.uniform(-amount, amount, (1, 1, 1))
    n2 = np.random.uniform(-amount2, amount2, (1, 1, 1))
    #print(n.shape)
    
    result = hsv_img.astype(np.float32)
    result[:, :, 0:1] += n
    result[:, :, 2:3] += n2
    result[result[:,:,0:1]<0] += 179
    result[result[:,:,0:1]>179] -= 179
    result[result<0] = 0
    result[result>255] = 255
    result = result*cropMask
    
    result = result.astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    return result

def flip(img):
    result = img.copy()
    if np.random.uniform(0, 1) > 0.5:
        result = cv2.flip( result, 0 )
    if np.random.uniform(0, 1) > 0.5:
        result = cv2.flip( result, 1 )
    return result


def generateAugmentedImage(crops, backgrounds):

    bgIndex = np.random.randint(0, backgrounds.shape[0])

    # resulting image resolution (darknet default)
    dx = 640
    dy = 480

    im = backgrounds[bgIndex, :, :, :]

    #print(im.shape)

    h = im.shape[0]
    w = im.shape[1]
    
    x = random.randint(0, w-dx)
    y = random.randint(0, h-dy)
    #print("Cropping {}: {},{} -> {},{}".format(file, x,y, x+dx, y+dy))
    bgCrop = im[y:y+dy, x:x+dx, :].copy()


    ratioX = np.random.uniform(0.3, 1.0)
    ratioY = np.random.uniform(0.3, 1.0)

    # hardcoded label list, must coincide with the prefix of video file names
    cropsLabels = []
    classNames = ['bowl', 'cup', 'fork', 'knife', 'napkin', 'plate', 'spoon']
    for i in range(len(crops)):
        label = crops[i][0]
        cropsLabels.append(label)

    classWeights = []
    clsSum = 0
    for i in range(len(cropsLabels)):
        label = crops[i][0]
        p = 1.0/cropsLabels.count(label)/len(classNames)
        classWeights.append(p)
        clsSum += p
    
    
    # print(cropsLabels)
    # print(classWeights)

    bgCrop = flip(bgCrop)

    labelList = []

    for i in range(np.random.randint(1, 7)):


        #objIndex = np.random.randint(0, len(crops))

        objIndex = np.random.choice(len(crops), 1, p=classWeights)[0]

        label = crops[objIndex][0]
        cropIndex = np.random.randint(0, len(crops[objIndex][1]))
        crop = crops[objIndex][1][cropIndex].copy()

        #print(crop.shape)


        
        #print("Cropping {}: {},{} -> {},{}".format(file, x,y, x+dx, y+dy))
        
        # --------------- #
        # hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # hsv[:,:,1:2] = hsv[:,:,0:1]
        # hsv[:,:,2:3] = hsv[:,:,0:1]
        # #hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow("Before", hsv)
        
        crop = resize(crop)
        crop = rotate(crop)
        crop = color(crop)
        crop = flip(crop)

        # hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # hsv[:,:,1:2] = hsv[:,:,0:1]
        # hsv[:,:,2:3] = hsv[:,:,0:1]
        # #hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow("After", hsv)

        #cv2.waitKey(5000)
        # --------------- #

        dx = crop.shape[1]
        dy = crop.shape[0]

        #print(bgCrop.shape)

        h = bgCrop.shape[0]
        w = bgCrop.shape[1]
        
        #print(w-dx-1, h-dy-1)
        x = random.randint(0, w-dx)
        y = random.randint(0, h-dy)

        blendRatio = np.random.uniform(0.0, 0.2)
        cropMask = np.sum(crop.copy(), axis=2)
        cropMask[cropMask > 0] = 1
        cropMask = cropMask.reshape(cropMask.shape[0], cropMask.shape[1], 1)
        bgCrop[y:y+dy, x:x+dx, :] = bgCrop[y:y+dy, x:x+dx, :]*(1-cropMask) + bgCrop[y:y+dy, x:x+dx, :]*cropMask*blendRatio + cropMask*crop*(1-blendRatio)

        labelList.append([classNames.index(label), (x+dx/2.0)/bgCrop.shape[1], (y+dy/2.0)/bgCrop.shape[0], 1.0*dx/bgCrop.shape[1], 1.0*dy/bgCrop.shape[0]])

    bgCrop = color2(bgCrop)
    bgCrop = noise(bgCrop)

    # cv2.imshow("Image", bgCrop)
    # cv2.waitKey(100)

    return bgCrop, labelList


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data generator.')
    parser.add_argument('--trainSize', type=int, default=100, help='training set size, default 100')
    parser.add_argument('--validSize', type=int, default=10, help='validation set size, default 10')
    parser.add_argument('--testSize', type=int, default=10, help='test set size, default 10')
    #parser.add_argument('--display', metavar='D', type=int, default=1, help='show cropped objects (0==False, 1==True), default 1')
    parser.add_argument('--verbose', metavar='V', type=int, default=1, help='output text to console (0==False, 1==True), default 1')
    args = parser.parse_args()

    crops = []
    
    for root, dirs, files in os.walk("../generated_crop_data"):  
        for filename in files:
            #print(filename)

            label = filename.split("_")[0]

            if args.verbose:
                print(label)
            
            with open('../generated_crop_data/'+filename, 'rb') as handle:
                crops.append([label, pickle.load(handle)])

    # hack count files
    nrBg = 0
    for root, dirs, files in os.walk("../dataset_backgrounds"):
        for filename in files:
            nrBg += 1

    # background images are currently 1600x1200
    backgrounds = np.zeros((nrBg, 1200, 1600, 3), dtype=np.uint8)
    bgIndex = 0
    for root, dirs, files in os.walk("../dataset_backgrounds"):
        for filename in files:
            im = cv2.imread("../dataset_backgrounds/"+filename, cv2.IMREAD_COLOR)

            backgrounds[bgIndex,:,:,:] = im
            bgIndex += 1
            # cv2.imshow("Image", im)
            # cv2.waitKey(1)


    sets = ['train', 'valid', 'test']
    set_size = [args.trainSize, args.validSize, args.testSize]

    # safety check to avoid replacing previous data !
    if not os.path.exists('../generated_darknet_data/'):
        # generate folders
        os.makedirs('../generated_darknet_data/')
        for s in sets:
            if not os.path.exists('../generated_darknet_data/'+s):
                os.makedirs('../generated_darknet_data/'+s)
                os.makedirs('../generated_darknet_data/'+s+'/images')
                os.makedirs('../generated_darknet_data/'+s+'/labels')

        for si in range(len(sets)):
            file_list = open("../generated_darknet_data/"+sets[si]+".txt", "w")
            for i in range(set_size[si]): #range(600000):
                bgCrop, labels = generateAugmentedImage(crops, backgrounds)

                cv2.imwrite("../generated_darknet_data/"+sets[si]+"/images/img_%d.jpg" % i, bgCrop)
                with open("../generated_darknet_data/"+sets[si]+"/labels/img_%d.txt" % i, "w") as text_file:
                    for j in range(len(labels)):
                        line = str(labels[j][0])
                        for k in range(1, len(labels[j])):
                            line += " "+str(labels[j][k])
                        text_file.write(line+"\n")
                file_list.write("../generated_darknet_data/"+sets[si]+"/images/img_%d.jpg\n" % i)
                #print(labels)
            file_list.close()
    else:
        print('------------------------------------------------------------------')
        print('| SANITY CHECK: ../generated_darknet_data folder already exists! |')
        print('|         Please move or remove it to generate new data.         |')
        print('------------------------------------------------------------------')



    # while True:
    #     pass
            
            # print(image.shape)
            # count = 0
            # success = True

            # cropList = []
            # while success:
            #     #cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file

            #     if count % 2 == 0:
            #         img, mask = processImage(image)

            #         kernel1 = np.ones((4,4), np.uint8)
            #         kernel2 = np.ones((7,7), np.uint8)

            #         mask = cv2.erode(mask, kernel1, iterations=2)

            #         mask = cv2.dilate(mask, kernel2, iterations=2)

            #         crop, cmask = cropObject(img, mask)

            #         maxSize = max(crop.shape[0], crop.shape[1])

            #         ratio = 640.0/maxSize

            #         small_crop = cv2.resize(crop, (int(crop.shape[1]*ratio), int(crop.shape[0]*ratio)))
            #         small_cmask = cv2.resize(cmask, (int(cmask.shape[1]*ratio), int(cmask.shape[0]*ratio)))

            #         cropList.append(small_crop)

                    
            #         cv2.imshow("Image", small_crop)
            #         cv2.waitKey(1)


            #     success,image = vidcap.read()
            #     print 'Read a new frame: ', success, count
            #     count += 1

            # with open('./crop_data/'+filename+'_cropped.pickle', 'wb') as handle:
            #     pickle.dump(cropList, handle, protocol=pickle.HIGHEST_PROTOCOL)