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
2. [Instructions](#2.0)
3. [Dataset Download](#3.0)
4. [TO-DO List](#4.0)

---
### 1.0 - Authors

- M.Sc. Caio Viturino* - [[Lattes](http://lattes.cnpq.br/4355017524299952)] [[Linkedin](https://www.linkedin.com/in/engcaiobarros/)] - engcaiobarros@gmail.com
- M.Sc. Kleber de Lima Santana Filho** - [[Lattes](http://lattes.cnpq.br/3942046874020315)] [[Linkedin](https://www.linkedin.com/in/engkleberfilho/)] - engkleberf@gmail.com
- M.Sc. Daniel M. de Oliveira* - [[Linkedin](https://www.linkedin.com/in/daniel-moura-de-oliveira-9b6754120/)] - danielmoura@ufba.br 
- Prof. Dr. André Gustavo Scolari Conceição* - [[Lattes](http://lattes.cnpq.br/6840685961007897)] - andre.gustavo@ufba.br

*LaR - Laboratório de Robótica, Departamento de Engenharia Elétrica e de Computação, Universidade Federal da Bahia, Salvador, Brasil

**PPGM - Programa de Pós-Graduação em Mecatrônica, Universidade Federal da Bahia, Salvador, Brasil.

---

### 2.0 - Instructions

> For this project to work please install `Tensorflow 2.x` and `Python: 3.8`. It may need some code adaptations to work in previous versions.

This repository gives you the tools to generate TFRecord files (train, validation, and test files) from images.

> All the following scripts may need small modifications in order to fit your data.

- Please configure the file `config_files/config.json` and `label_map.json` to fit your images and files features and path. You just need to consider the repository folder as a root to reference the paths. The python code takes care of the rest.
- You should create a folder called `images` and put the [dataset images](#3.0) into this folder according to the class folder. Ex: `images/bar_clamp` or `images/gear_box` 
- If you want to convert PASCAL VOC xml files to csv, you should create a folder `xml/PASCAL VOC`, put all your xml files into this folder and run the script `etc/xml_to_csv.py`. It may need small modifications to fit your data.
- You should put your csv file into the `csv` folder.
  - Your csv file must be in the following format: 
    ```sh 
    "image","xmin","ymin","xmax","ymax","label"
    "001.jpg",1843.5669909434714,1134.8309303950339,2521.393150378185,1610.1179645130937,"bar_clamp"
    ```
- Split your data by using the script `utils/split_data.py`. It will generate train, validation, and test csv files into the `data` folder.
- Generate the TFRecord file by using the script `utils/generate_tfrecord.py`. It will generate TFRecord files into the `data` folder.
- View your data by using the script `utils/view_record.py`. It will plot all the images and the bounding boxes from the TFRecord using OpenCV.

---

### 3.0 - Dataset Download

Training objects:

[Bar clamp](https://drive.google.com/file/d/1F9DrgFgdt-cfT-cjSImfLCjmFoc4V3SN/view?usp=sharing)

[Gear box](https://drive.google.com/file/d/1JtUpxdmQf-T1sV6RTKfxf-LrZf1Cl-W8/view?usp=sharing)

---

### 4.0 - TO-DO List

#### 4.1 - Detecção de objetos

![25$](https://progress-bar.dev/25) - Tirar 160 fotos de cada peça (são 8 peças no total) - Responsável: Caio

![25$](https://progress-bar.dev/25) - Fazer o annotation das fotos usando o software [LabelImg](https://github.com/tzutalin/labelImg) - Responsáveis: Caio, Kleber e Daniel

![100$](https://progress-bar.dev/25) - Criar um script para TensorFlow para carregamento e visualização do dataset - Responsável: Caio

![0$](https://progress-bar.dev/0) - Treinar a SSD512 com a ResNet50

![0$](https://progress-bar.dev/0) - Treinar a SSD512 com a VGG16

![0$](https://progress-bar.dev/0) - Treinar a SSD300 com a ResNet50

![0$](https://progress-bar.dev/0) - Treinar a SSD300 com a VGG16

![0$](https://progress-bar.dev/0) - Treinar a Faster R-CNN ResNet50

![0$](https://progress-bar.dev/0) - Definir quais serão os pré-processamentos aplicados nas imagens (variação de cor, brilho, random crop, zoom, etc) - Nesse passo a gente deve padronizar o código de pré-processamento para usar nos treinamentos de todas as redes - Responsáveis: Caio, Kleber e Daniel

#### 4.2 - Parte prática


![100$](https://progress-bar.dev/100) - Imprimir os "Adversarial Objects" do Dex-Net 2.0 - Responsável: Caio

![0$](https://progress-bar.dev/0) - Atualizar a simulação no Gazebo, colocando as peças impressas para testar a performance da rede em ambientes virtuais.

![0$](https://progress-bar.dev/0) - Adicionar um limitador de altura para o grasp com o objetivo de evitar acidentes

![0$](https://progress-bar.dev/0) - Fazer diversos testes no laboratório com o sistema. Serão realizadas 20 preensões por objeto. 

![0$](https://progress-bar.dev/0) - Medir a performance de todas as redes em "ms" ao invés de "FPS" (Adotado por muitos autores na área de grasping). Talvez tenhamos que escolher o hardware de um dos autores para padronizar os dados de performance. 

![0$](https://progress-bar.dev/0) - Gravar um vídeo do experimento prático

![0$](https://progress-bar.dev/0) - Gravar um vídeo do experimento simulado

#### 4.3 - Parte escrita


![0$](https://progress-bar.dev/0) - Definir um journal para publicação - Responsável: Todos

![0$](https://progress-bar.dev/0) - Desenvolver a base teórica para o conteúdo apresentado - Responsável: Todos

![0$](https://progress-bar.dev/0) - Iniciar a escrita do artigo - Responsável: Todos

