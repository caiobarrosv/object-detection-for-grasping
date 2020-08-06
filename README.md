## object-detection-for-grasping
Repository dedicated to build a set Convolutional Neural Networks models to detect objects in order to perform a selective grasp.
This project is part if a bigger grasping pipeline firstly implemented in this [repository](https://github.com/lar-deeufba/ssggcnn_ur5_grasping) by the Authors cited below.

This repository is divided in branches for each framework implementation. The `master` branch refers to the `GluonCV (MXNet)` implementation. The `Tensorflow` branch refers to the `Tensorflow 2.0` implementation (both are under development).

<!--<p align="center">
<a href="https://youtu.be/aJ39MruDdLo" target="_blank">
<img src="" width="600">
</p>
</a>-->

### Contents
1. [Authors](#1.0)
2. [Requirements](#2.0)
3. [Instructions](#3.0)
4. [Dataset Download](#4.0)

---
<a name="1.0"></a>
### 1.0 - Authors

- M.Sc. Caio Viturino* - [[Lattes](http://lattes.cnpq.br/4355017524299952)] [[Linkedin](https://www.linkedin.com/in/engcaiobarros/)] - engcaiobarros@gmail.com
- M.Sc. Kleber de Lima Santana Filho** - [[Lattes](http://lattes.cnpq.br/3942046874020315)] [[Linkedin](https://www.linkedin.com/in/engkleberfilho/)] - engkleberf@gmail.com
- M.Sc. Daniel M. de Oliveira* - [[Linkedin](https://www.linkedin.com/in/daniel-moura-de-oliveira-9b6754120/)] - danielmoura@ufba.br 
- Prof. Dr. André Gustavo Scolari Conceição* - [[Lattes](http://lattes.cnpq.br/6840685961007897)] - andre.gustavo@ufba.br

*LaR - Laboratório de Robótica, Departamento de Engenharia Elétrica e de Computação, Universidade Federal da Bahia, Salvador, Brasil

**PPGM - Programa de Pós-Graduação em Mecatrônica, Universidade Federal da Bahia, Salvador, Brasil.

---
<a name="2.0"></a>
### 2.0 - Requirements

Please install the following:

- mxnet-cu101 1.5.0
- Cuda 10.1
- CuDNN 7.6.5

If you use conda, set up a new environment and activate it by using:

```
conda create --name object_detection_tf2
conda activate object_detection_tf2
```

If you want a shortcut to the required packages, install them by doing:
```
pip install -r requirements.txt
```

If you would like to see your training parameters in tensorboard, please follow the procedures stated in `https://github.com/awslabs/mxboard`.
use the following commands to se the data in the web-browser:
```
tensorboard --logdir=./logs --host=127.0.0.1 --port=8888
```
---
<a name="3.0"></a>
### 3.0 - Instructions

This repository gives you the tools to generate record files (train, validation, and test files) from images and train models provided by GluonCV.

> All the following scripts may need small modifications in order to fit your data.

- **How to change the file paths in all the files without modifying them individually?**
  - Please configure the file `config_files/config.json` to fit your images and files. You just need to configure the relative path.

- **How to organize my images?**
  - You should create a folder called `images` and put all the [dataset images](#3.0) into this folder. (Note: please, put all the images into this folder and **do not divide by class**). 
  - If you have a .rec file, just put this file in the `data` folder and configure the path in config.json.

- **How can I check my csv files before generating the lst file (default in MXNet)?**
  - If you want to view the images and the bounding boxes pointed in the csv file, please run the script `utils/view_csv_files.py`.

- **Can I modify the images sizes before generating a new lst file?**
  - If you want to resize your images and save these images and a new csv file containing the resized bounding boxes and images sizes in a new folder, please run the script `utils/resize_images_csv.py` 

- **How is the csv format used in this project?**
  - Your csv file must be in the following format: 
    ```sh 
    image,xmin,ymin,xmax,ymax,label
    000.jpg,463,273,635,450,bar_clamp
    ```

- **How to divide my csv file in train/val/test csv files and shuffle them?** 
  - Split your data by using the script `utils/split_data.py`. It will generate train, validation, and test csv files into the `csv` folder.
  - If you just want to shuffle your data, set the validation or train percentage to 100%.

- **How to generate a lst and record file for the GluonCV?**
  - Please use the script `utils/prepate_dataset.py` for generating train/val lst files and the record files. 
  - The record files will be saved in the same folder of the lst files.

- **How to generate a record file from a lst file?**
  - You do not need to run the script below, just use the `utils/prepate_dataset.py` script. It will generate the record files for you automatically. But if you want to use the script that converts the lst files into record files separately, please follow the procedures stated below:
    - Use the `im2rec.py` script provided in the root folder and type the following command:
      ```sh
      python im2rec.py lst_file_name relative_root_to_images --pass-through --pack-label
      ```
      
      Since the images path are included in the lst file when generating the lst file, please set the relative_root_to_images as the root folder [.]

      Example:
      ```sh
      python im2rec.py data/train.lst . --pass-through --pack-label
      python im2rec.py data/val.lst . --pass-through --pack-label
      ```
    
- **How can I make sure that my record file was correctly generated?**
  - View your data by using the script `utils/view_record_mxnet.py`. It will plot all the images and the bounding boxes from the record using OpenCV.

- **What if I don't have my files in csv format?**
  - If you want to convert PASCAL VOC xml files to csv, you should create a folder `xml/PASCAL VOC`, put all your xml files into this folder and run the script `etc/xml_to_csv.py`. It may need small modifications to fit your data.
  - Other parse scripts are under development

---
Folder structure:

Note that you should create some folders yourself or change the path configuration in the `config.json` file.

```bash
├───config_files      # Files path configuration
├───csv               # CSV files folder
├───data              # Record folder
├───etc               # Auxiliary code
├───images            # You should create this folder and put your images
├───utils             # Codes to manage the dataset
└───xml               # Your .xml files
    └───PASCAL VOC
```

---
<a name="4.0"></a>
### 4.0 - Dataset Download

This dataset contains the gear box and bar clamp images.
A bigger dataset is under development.

Bouding box format: xmin, ymin, xmax, ymax

- [Dataset 1](https://drive.google.com/file/d/1IrBlQRCX4731ISnXCqnuGpLrakFWPPbB/view?usp=sharing) 
  - Image resolution: (756, 1008, 3) # (height, width, channel)
  - Type of the files: .jpeg
  - Number of images: 320
  - Classes: Bar clamp and gear box
  - In this dataset, the train and validation files have 1 classe per image with only 1 type of background
  - Ground truth: 1 ground truth per image (only 1 part is in the image)
<p align="center">Example of train and validation image:</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/28100951/89105060-65b40900-d3f4-11ea-876a-cac81d3b1ec1.png" width="200"/>
</p>


- [Dataset 2](https://drive.google.com/file/d/1zf8WPfhgEjUtlamjo58V3URlfXsPl-U6/view?usp=sharing) 
  - Image resolution: (300, 300, 3) # (height, width, channel)
  - Type of the files: .jpeg
  - Number of images: 320
  - Classes: Bar clamp and gear box
  - In this dataset, the train and validation files have 1 classe per image with only 1 type of background
  - Ground truth: 1 ground truth per image (only 1 part is in the image)
<p align="center">Example of train and validation image:</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/28100951/89105095-998f2e80-d3f4-11ea-9c0e-3c0dcdc494cf.png" width="190">
</p>

- [Dataset - Teste 4](https://drive.google.com/file/d/15L-iOGPgF5al5sHx3rLiIF_F2Hs3iQjd/view?usp=sharing) 
  - Image resolution: (300, 300, 3) # (height, width, channel) 
  - Type of the files: .rec
  - Training images: 320 train
  - Validation images: 64 val
  - Classes: Bar clamp and gear box
  - In this dataset, the train images have all the classes together with the same background as well as the validation files with the same background
  - Ground truths: 2 ground truths per image (all the 8 parts are in the image)
<p align="center">Example of train image:</p>
<p align="center"> 
<img src="https://user-images.githubusercontent.com/28100951/89105110-bfb4ce80-d3f4-11ea-928e-c1606922ee41.png" width="160" title="train image">
</p>
<p align="center">Example of validation image:</p>
<p align="center"> 
<img src="https://user-images.githubusercontent.com/28100951/89105112-c0e5fb80-d3f4-11ea-9a87-77b326f586b0.png" width="160">
</p> 


- [Dataset - Teste 5](https://drive.google.com/file/d/1AT0_UCA-sP9Mt8z-KuPkPSmVKocC84XZ/view?usp=sharing) 
  - Image resolution: (300, 300, 3) # (height, width, channel)
  - Type of the files: .rec
  - Training images: 560 train
  - Validation images: 64 val
  - Classes: Bar clamp and gear box
  - The train images have all the classes together with different backgrounds as well as the validation files with the same background
  - Ground truths: 2 ground truths per image (all the 8 parts are in the image)
<p align="center">Example of train images:</p>
<p align="center"> 
<img src="https://user-images.githubusercontent.com/28100951/89105201-77e27700-d3f5-11ea-80e0-b6abbddc7afa.png"  width="400" title="train image">
</p> 

<p align="center">Example of validation image:</p>
<p align="center"> 
<img src="https://user-images.githubusercontent.com/28100951/89105225-bb3ce580-d3f5-11ea-9463-ee16d6c67484.png"  width="100" title="train image">
</p> 

- [Dataset - Teste 6](https://drive.google.com/file/d/1dEAJj7fm_lxxrBl0w8gEeSOg3AHCHxZ0/view?usp=sharing) 
  - Image resolution: (300, 300, 3) # (height, width, channel)
  - Type of the files: .rec
  - Training images: 500 train
  - Validation images: 124 val
  - Classes: Bar clamp and gear box
  - The train images have all the classes together with different backgrounds 
    - Ground truths: 2 ground truths per image (all the 8 parts are in the image)
  - The validation files also have different backgrounds. 
    - Ground truths: 1 ground truths in the images like the left image
    - Ground truths: 2 ground truths per image in the images like the right image
  - Remember to set the post_nms to be equal the max number of classes of the validation dataset. In this dataset, we can have 2 classes by image:
    - `self.net.set_nms(nms_thresh=nms_threshold, nms_topk=200, post_nms=2)`
  
<p align="center">Example of train images:</p>
<p align="center"> 
<img src="https://user-images.githubusercontent.com/28100951/89105914-b5e29980-d3fb-11ea-8758-067881325f70.png"  width="300" title="train image">
</p> 

<p align="center">Example of validation images:</p>
<p align="center"> 
<img src="https://user-images.githubusercontent.com/28100951/89105917-b713c680-d3fb-11ea-9d31-6d402c875b6d.png"  width="180" title="train image">
</p> 

- Dataset - Teste 7
  - Train and validation images (real size): https://drive.google.com/file/d/1a1JFObOwLAc2mb7jvIjkpynpfAwzaeV7/view?usp=sharing
    - Image resolution: Some images are (800, 800, 3) # (height, width, channel) and others (1000, 1000, 3) # (height, width, channel)
    - Type of the files: .rec
  - Train and validation images (resized): https://drive.google.com/file/d/1a1JFObOwLAc2mb7jvIjkpynpfAwzaeV7/view?usp=sharing
    - Image resolution: (300, 300, 3) # (height, width, channel)
    - Type of the files: .rec
  - Training images: 500 train
  - Validation images: 124 val
  - Classes: Bar clamp, gear box, vase, part_1, part_3, nozzle, pawn, and turbine_housing
  - The train and validation images have the ground truths for all the classes
  - Remember to set the post_nms to be equal the max number of classes of the validation dataset. In this dataset, we can have 2 classes by image:
    - `self.net.set_nms(nms_thresh=nms_threshold, nms_topk=200, post_nms=2)`
  - Obs.: The only difference between dataset 7 and dataset 6 is that the dataset 7 contains the ground truths for all the classes 

- [Video files for test - 1, 2, 3 and 4](https://drive.google.com/file/d/1pSBK4ZSbuQ6eEz-dKDKgM2LBzYU0E8Tb/view?usp=sharing) 
  - Video resolution: (1440, 1440, 3) # (height, width, channel)
  - Type of the files: .mp4
  - Videos: 4 files
  - Classes: All classes together

<p align="center">Screenshot of one of the videos:</p>
<p align="center"> 
<img src="https://user-images.githubusercontent.com/28100951/89115443-9fb9f500-d45e-11ea-8ffb-e6fd14ac070c.png"  width="150" title="train image">
</p> 