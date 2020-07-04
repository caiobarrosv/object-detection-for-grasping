# adapted from: 
# https://gist.github.com/rotemtam/88d9a4efae243fc77ed4a0f9917c8f6c

import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
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

            value = (root.find('filename').text,
                    #  int(root.find('size')[0].text),
                    #  int(root.find('size')[1].text),
                     xmin,
                     ymin,
                     xmax,
                     ymax,
                     label,
                     )
            xml_list.append(value)
    column_name = ['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
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