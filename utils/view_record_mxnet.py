from gluoncv.data import recordio
import numpy as np
import cv2
import dataset_commons

data_common = dataset_commons.get_dataset_files()
record_train_path = data_common['record_train_path']
record_val_path = data_common['record_val_path']
classes_keys = [key for key in data_common['classes']]

def check_record_file(record_path):
    '''
    This function plots the image recorded in a .record file

    Arguments:
        record_path (str): the relative path of the project root folder
    '''
    record_file = recordio.detection.RecordFileDetection(record_path)
    for img, labels in record_file: 
        for label in labels:
            [xmin, ymin, xmax, ymax] = [int(label) for label in label[0:-1]]
            class_id = int(label[-1])
            class_name = classes_keys[class_id]

            if not isinstance(img, np.ndarray):
                img = img.asnumpy()
                img = img.astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # OpenCV uses BGR order

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
            cv2.putText(img, class_name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

            cv2.startWindowThread()
        
        cv2.imshow('img', img)
        a = cv2.waitKey(0) # close window when ESC is pressed
        if a == 27:
            break
        cv2.destroyWindow('img')
    
def main():
    print("\nPlease configure the record files path in config.json before running the next command.")    
    a = int(input("Choose to visualize the train.rec file [option: 1] or val.rec file [option: 2]: "))

    if a == 1:
        record_path = record_train_path
    elif a == 2:
        record_path = record_val_path
    else:
        print("Please choose the right option")        

    check_record_file(record_path)

if __name__ == "__main__":
    main()