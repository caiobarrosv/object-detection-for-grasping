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

---

### 3.0 - Instructions

This repository gives you the tools to generate record files (train, validation, and test files) from images and train models provided by GluonCV.

> All the following scripts may need small modifications in order to fit your data.

- **How to change the file paths in all the files without modifying them individually?**
  - Please configure the file `config_files/config.json` and `label_map.json` to fit your images and files features and path. You just need to consider the repository folder as a root to reference the paths. The python code takes care of the rest.

- **How to organize my images?**
  - You should create a folder called `images` and put the [dataset images](#3.0) into the folder. (Note: please, put all the images into this folder and **do not divide by class**)

- **How can I check my csv files before generating the lst file (default in MXNet)?**
  - If you want to view the images and the bounding boxes pointed in the csv file, please run the script `utils/view_csv_files.py`.

- **Can I modify the images sizes before generating a new record file?**
  - If you want to resize your images and save these images and a new csv file containing the resized bounding boxes and images sizes in a new folder, please run the script `utils/resize_images_csv.py` 

- **Where to put my csv files?**
  - You should put your csv file into the `csv` folder.
  - Your csv file must be in the following format: 
    ```sh 
    image,xmin,ymin,xmax,ymax,label,height,width
    000.jpg,463,273,635,450,bar_clamp,756,1008
    ```

- **How to divide my csv file in train/val/test csv files and shuffle them?** 
  - Split your data by using the script `utils/split_data.py`. It will generate train, validation, and test csv files into the `csv` folder.

- **How to generate a lst file for the GluonCV?**
  - Please use the script `utils/prepate_dataset.py` for generating train/val lst files. The files will be saved in the data folder according to the config.json

- **How to generate a record file from a lst file?**
  - Use the `im2rec.py` script provided in the root folder and type the following command:
    ```sh
    python im2rec.py lst_file_name relative_root_to_images --pass-through --pack-label
    ```
    
    Since the images path are included in the lst file when generating the lst file, please set the relative_root_to_images as the root folder [.]

    Example:
    ```sh
    python im2rec.py data/train.lst . --pass-through --pack-label
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
