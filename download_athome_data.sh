#!/bin/sh

######################################################################################
#Set up RoboCup@home dataset

echo "-------------------------------------"
echo "Checking if I have dataset archive..."
echo "-------------------------------------"
if [ ! -f ./data.augmentation.athome.tar.gz ]; then
	echo "--------------------------------------------"
    echo "I don't have it. Downloading dataset arhive."
    echo "--------------------------------------------"
    wget https://www.dropbox.com/s/h1gy5qbhedevcao/data.augmentation.athome.tar.gz
else
    echo "----------------"
    echo "Already have it."
    echo "----------------"
fi

echo "-------------------------------------------"
echo "Checking if I have extracted the archive..."
echo "-------------------------------------------"
if [ ! -d ./dataset_videos ] || [ ! -d ./dataset_backgrounds ]; then
    echo "------------------------"
    echo "I din't. Extracting now."
    echo "------------------------"
    tar xvzf ./data.augmentation.athome.tar.gz
else
    echo "-----------------------------"
    echo "Already extracted it. Enjoy !"
    echo "-----------------------------"
fi

######################################################################################