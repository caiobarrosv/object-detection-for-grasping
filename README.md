## object-detection-for-grasping
Repository dedicated to build a set Convolutional Neural Networks models to detect objects in order to perform a selective grasp.
This project is part if a bigger grasping pipeline firstly implemented in this [repository](https://github.com/lar-deeufba/ssggcnn_ur5_grasping) by the Authors cited below.

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
### 1.0 - Authors

- M.Sc. Caio Viturino* - [[Lattes](http://lattes.cnpq.br/4355017524299952)] [[Linkedin](https://www.linkedin.com/in/engcaiobarros/)] - engcaiobarros@gmail.com
- M.Sc. Kleber de Lima Santana Filho** - [[Lattes](http://lattes.cnpq.br/3942046874020315)] [[Linkedin](https://www.linkedin.com/in/engkleberfilho/)] - engkleberf@gmail.com
- M.Sc. Daniel M. de Oliveira* - [[Linkedin](https://www.linkedin.com/in/daniel-moura-de-oliveira-9b6754120/)] - danielmoura@ufba.br 
- Prof. Dr. André Gustavo Scolari Conceição* - [[Lattes](http://lattes.cnpq.br/6840685961007897)] - andre.gustavo@ufba.br

*LaR - Laboratório de Robótica, Departamento de Engenharia Elétrica e de Computação, Universidade Federal da Bahia, Salvador, Brasil

**PPGM - Programa de Pós-Graduação em Mecatrônica, Universidade Federal da Bahia, Salvador, Brasil.

---

### 2.0 - Requirements

Please install the following:

- Cuda 10.1
- CuDNN 7.6.5

If you use conda, set up a new environment and activate it by using:

```
conda create --name object_detection_tf2
conda activate object_detection_tf2
```

Install this repository and the required packages
```
git clone https://github.com/caiobarrosv/object-detection-for-grasping
pip install -r requirements.txt
```

> **Important note:** This repository uses the code provided in `github.com/pierluigiferrari/ssd_keras` but you don't need to clone them, although it is a must to visit them, give a star, follow and read all the documentation because it is really well documented. The codes from this repository with small changes are given in `train_utils` folder.

---

### 3.0 - Instructions

> For this project to work please install `Tensorflow 2.x` and `Python: 3.8`. It may need some code adaptations to work in previous versions.

This repository gives you the tools to generate TFRecord files (train, validation, and test files) from images.

> All the following scripts may need small modifications in order to fit your data.

- Please configure the file `config_files/config.json` and `label_map.json` to fit your images and files features and path. You just need to consider the repository folder as a root to reference the paths. The python code takes care of the rest.
- You should create a folder called `images` and put the [dataset images](#3.0) into this folder according to the class folder. Ex: `images/bar_clamp` or `images/gear_box` 
- If you want to convert PASCAL VOC xml files to csv, you should create a folder `xml/PASCAL VOC`, put all your xml files into this folder and run the script `etc/xml_to_csv.py`. It may need small modifications to fit your data.
- If you want to view the images and the bounding boxes pointed in the csv file, please run the script `utils/view_csv_files.py`
- If you want to resize your images and save these images and a new csv file containing the resized bounding boxes and images sizes in a new folder, please run the script `utils/resize_images_csv.py` 
- You should put your csv file into the `csv` folder.
  - Your csv file must be in the following format: 
    ```sh 
    image,xmin,ymin,xmax,ymax,label,height,width
    000.jpg,463,273,635,450,bar_clamp,756,1008
    ```
- Split your data by using the script `utils/split_data.py`. It will generate train, validation, and test csv files into the `csv` folder.
- Generate the TFRecord file by using the script `utils/generate_tfrecord.py`. It will generate TFRecord files into the `data` folder.
- View your data by using the script `utils/view_record.py`. It will plot all the images and the bounding boxes from the TFRecord using OpenCV.
- If you want to use your dataset in hdf5 format and you have only the csv file, you can use the `utils/create_hdf5_dataset_from_csv.py`. This code was created by `github.com/pierluigiferrari`, and we have made some small changes to fit our dataset.

Folder structure:

Note that you should create some folders yourself or change the path configuration in the `config.json` file.

```bash
├───config_files      # Files path configuration
├───csv               # CSV files folder
├───data              # TFRecord folder
├───etc               # Auxiliary code
├───images            # You should create this folder and put your images
├───utils             # Codes to manage the dataset
└───xml               # Your .xml files
    └───PASCAL VOC
```

---

### 4.0 - Dataset Download

This dataset contains the gear box and bar clamp images.
A bigger dataset is under development.

Image resolution: (756, 1008) # (height, width)
Bouding box format: xmin, ymin, xmax, ymax

[Dataset](https://drive.google.com/file/d/1IrBlQRCX4731ISnXCqnuGpLrakFWPPbB/view?usp=sharing)
