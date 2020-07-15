import os
import csv
from xml.dom import minidom
import xml.etree.ElementTree as ET

'''
Script created by Daniel Oliveira
it reads a csv file and creates a PASCAL VOC format dataset file
'''

def read_csv(csv_file):
    image = []
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    label = []
    width = []
    height = []
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image.append(row['image'])
            xmin.append(row['xmin'])
            xmax.append(row['xmax'])
            ymin.append(row['ymin'])
            ymax.append(row['ymax'])
            label.append(row['label'])
            width.append(row['width'])
            height.append(row['height'])
    return image,xmin,xmax,ymin,ymax,label,width,height

def convert_to_vocxml(image,xmin,xmax,ymin,ymax,label,width,height):
    #Cria os labels utilizados no XML do VOC
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation,'folder')
    filename = ET.SubElement(annotation,'filename')
    path = ET.SubElement(annotation,'path')
    source = ET.SubElement(annotation,'source')
    size = ET.SubElement(annotation,'size')
    segmented = ET.SubElement(annotation,'segmented')
    object = ET.SubElement(annotation,'object')

    database = ET.SubElement(source,'database')
    width_xml = ET.SubElement(size,"width")
    height_xml = ET.SubElement(size,"height")
    depth = ET.SubElement(size,"depth")

    nome = ET.SubElement(object,'name')
    pose = ET.SubElement(object,'pose')
    truncated = ET.SubElement(object,'truncated')
    difficult = ET.SubElement(object,'difficult')
    bndbox = ET.SubElement(object,'bndbox')

    xmin_xml = ET.SubElement(bndbox,'xmin')
    ymin_xml = ET.SubElement(bndbox,'ymin')
    xmax_xml = ET.SubElement(bndbox,'xmax')
    ymax_xml = ET.SubElement(bndbox,'ymax')

    #Coloca os valores do csv no XML
    size = len(image)
    for i in range(size):
        folder.text = label[i]
        filename.text = image[i]
        path.text ='/your/path/' + image[i]
        database.text = 'Unknown'
        width_xml.text = width[i]
        height_xml.text = height[i]
        depth.text = '3'
        segmented.text = '0'
        nome.text = label[i]
        pose.text = "Unspecified"
        truncated.text = '0'
        difficult.text = '0'
        xmax_xml.text = xmax[i]
        ymax_xml.text = ymax[i]
        xmin_xml.text = xmin[i]
        ymin_xml.text = ymin[i]

        #Cria as pastas e o arquivo XML
        folder_xml = label[i] + '/'
        file = image[i].replace('.jpg','')
        path_file = folder_xml + file + ".xml"
        if not os.path.exists(folder_xml):
            os.makedirs(folder_xml)
        xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
        with open(path_file, "w") as f:
            f.write(xmlstr)


def main():
    csv_file = "adversarial_dataset_converted.csv"
    [image,xmin,xmax,ymin,ymax,label,width,height]=read_csv(csv_file)
    convert_to_vocxml(image,xmin,xmax,ymin,ymax,label,width,height)


main()