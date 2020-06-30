## object-detection-for-grasping
Repository dedicated to build a set Convolutional Neural Networks models to detect objects in order to perform a selective grasp.
This project is part if a bigger grasping pipeline firstly implemented in this [repository](https://github.com/lar-deeufba/ssggcnn_ur5_grasping) by the Authors cited below.

<!--<p align="center">
<a href="https://youtu.be/aJ39MruDdLo" target="_blank">
<img src="" width="600">
</p>
</a>-->

<a id="top"></a>
### Contents
1. [Authors](#1.0)
2. [Required Packages](#2.0)
3. [TO-DO List](#3.0)
------------
<a name="1.0"></a>
### 1.0 - Authors

- M.Sc. Caio Viturino* - [[Lattes](http://lattes.cnpq.br/4355017524299952)] [[Linkedin](https://www.linkedin.com/in/engcaiobarros/)] - engcaiobarros@gmail.com
- M.Sc. Kleber de Lima Santana Filho** - [[Lattes](http://lattes.cnpq.br/3942046874020315)] [[Linkedin](https://www.linkedin.com/in/engkleberfilho/)] - engkleberf@gmail.com
- M.Sc. Daniel M. de Oliveira* - [[Linkedin](https://www.linkedin.com/in/daniel-moura-de-oliveira-9b6754120/)] - danielmoura@ufba.br 
- Prof. Dr. André Gustavo Scolari Conceição* - [[Lattes](http://lattes.cnpq.br/6840685961007897)] - andre.gustavo@ufba.br

*LaR - Laboratório de Robótica, Departamento de Engenharia Elétrica e de Computação, Universidade Federal da Bahia, Salvador, Brasil

**PPGM - Programa de Pós-Graduação em Mecatrônica, Universidade Federal da Bahia, Salvador, Brasil.

<a name="2.0"></a>
### 2.0 - Required Packages

In discussion.

<a name="3.0"></a>
### 3.0 - TO-DO List

#### 3.1 - Detecção de objetos

![25$](https://progress-bar.dev/25) - Tirar 160 fotos de cada peça (são 8 peças no total) - Responsável: Caio

![25$](https://progress-bar.dev/25) - Fazer o annotation das fotos usando o software [LabelImg](https://github.com/tzutalin/labelImg) - Responsáveis: Caio, Kleber e Daniel

![0$](https://progress-bar.dev/0) - Treinar a SSD512 com a ResNet50

![0$](https://progress-bar.dev/0) - Treinar a SSD512 com a VGG16

![0$](https://progress-bar.dev/0) - Treinar a SSD300 com a ResNet50

![0$](https://progress-bar.dev/0) - Treinar a SSD300 com a VGG16

![0$](https://progress-bar.dev/0) - Treinar a Faster R-CNN ResNet50

![0$](https://progress-bar.dev/0) - Definir quais serão os pré-processamentos aplicados nas imagens (variação de cor, brilho, random crop, zoom, etc) - Nesse passo a gente deve padronizar o código de pré-processamento para usar nos treinamentos de todas as redes - Responsáveis: Caio, Kleber e Daniel

#### 3.2 - Parte prática


![100$](https://progress-bar.dev/100) - Imprimir os "Adversarial Objects" do Dex-Net 2.0 - Responsável: Caio

![0$](https://progress-bar.dev/0) - Atualizar a simulação no Gazebo, colocando as peças impressas para testar a performance da rede em ambientes virtuais.

![0$](https://progress-bar.dev/0) - Adicionar um limitador de altura para o grasp com o objetivo de evitar acidentes

![0$](https://progress-bar.dev/0) - Fazer diversos testes no laboratório com o sistema. Serão realizadas 20 preensões por objeto. 

![0$](https://progress-bar.dev/0) - Gravar um vídeo do experimento prático

![0$](https://progress-bar.dev/0) - Gravar um vídeo do experimento simulado

#### 3.3 - Parte escrita


![0$](https://progress-bar.dev/0) - Definir um journal para publicação - Responsável: Todos

![0$](https://progress-bar.dev/0) - Desenvolver a base teórica para o conteúdo apresentado - Responsável: Todos

![0$](https://progress-bar.dev/0) - Iniciar a escrita do artigo - Responsável: Todos

