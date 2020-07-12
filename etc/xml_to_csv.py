# adapted from: 
# https://gist.github.com/rotemtam/88d9a4efae243fc77ed4a0f9917c8f6c

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import cv2

def xml_to_csv(xml_path):
    xml_list = []
    i = 0
    for xml_file in glob.glob(xml_path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for member in root.findall('object'):
            bbx = member.find('bndbox')
            xmin = int(bbx.find('xmin').text)
            ymin = int(bbx.find('ymin').text)
            xmax = int(bbx.find('xmax').text)
            ymax = int(bbx.find('ymax').text)
            label = member.find('name').text
            
            if label == 'bar clamp':
                label = 'bar_clamp'
            if label == 'gear box':
                label = 'gear_box'

            filename = root.find('filename').text
            images_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'images', filename))
            print(images_path)
            img = cv2.imread(images_path)

            height, width, channels = img.shape # (4032, 3024, 3)
            print('antes: ', height, width, channels)

            value = (filename,
                     xmin,
                     ymin,
                     xmax,
                     ymax,
                     label,
                     str(height),
                     str(width)
                     )   
            xml_list.append(value)

    column_name = ['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'height', 'width']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
    xml_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'xml/PASCAL VOC'))
    xml_path_save = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'csv/adversarial_dataset_converted.csv'))
    xml_df = xml_to_csv(xml_path)
    xml_df.to_csv(xml_path_save, index=None)
    print('Successfully converted xml to csv.')

if __name__ == "__main__":
    main()