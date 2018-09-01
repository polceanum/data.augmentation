# Data augmentation scripts
Toolchain for data augmentation in Python, from video cropping to image generation.

# Get files

```
#!bash
git clone https://github.com/polceanum/data.augmentation.git
cd data.augmentation
```

# Download RoboCup@Home data

```
./download_athome_data.sh
```

# Data cleaner
Script for loading all files from ``dataset_videos`` and cropping based on background color. Creates the folder ``generated_crop_data`` with pickles for each video file.

```
cd scripts
python3 ./dataClean.py
```

# Data generator
Script for loading backgrounds from ``dataset_backgrounds`` and cropped objects from ``generated_crop_data`` and generating train, validation and test datasets in the format for darknet, under ``generated_darknet_data``.

```
cd scripts
python3 ./dataGen.py
```

# Current issues
* due to currently chosen background color in videos, cropped objects still have grayish borders, which leads to low quality generated images
* dataClean.py has color mask hardcoded
* dataGen.py is still mostly hardcoded and must be modified to configure generation rules
