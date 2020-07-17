from gluoncv.data import recordio
from gluoncv.utils import viz
import numpy as np
import cv2
from matplotlib import pyplot as plt

def check_record_file(record_path):
    '''
    This function plots the image recorded in a .record file

    Arguments:
        record_path (str): the relative path of the project root folder
    '''
    record_file = recordio.detection.RecordFileDetection(record_path)
    for img, label in record_file: 
        print(label)

        [xmin, ymin, xmax, ymax] = label[0][0:-1]
        class_id = label[0][-1]

        img = img.asnumpy()
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # OpenCV uses BGR order

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
        cv2.putText(img, 'label: ' + str(class_id), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

        cv2.startWindowThread()
        cv2.imshow('img', img)
        a = cv2.waitKey(0) # close window when ESC is pressed
        if a == 27:
            break
        cv2.destroyWindow('img')
    
def main():
    record_path = 'data/train.rec'
    check_record_file(record_path)

if __name__ == "__main__":
    main()