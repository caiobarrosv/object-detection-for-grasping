import cv2
import glob

'''
This script generates a new dataset from a video file.
You can choose the final image resolution in the main function.
'''

def image_extractor(targetSize, target_folder, source_files):
    '''
    Extract frames from videos and save them into .jpg files in the target folder.

    Arguments:
        targetSize (tuple, default : (1000, 1000)): Target images resolution
        target_folder (str) : absolute folder path
        source_files (list) : list of the video's absolute paths
    '''
    img_number=0
    for file in source_files:
        print(file)
        # Opens the Video file
        cap = cv2.VideoCapture(file)

        # Number of frames in the video
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Number of steps / photos extracted
        number_of_photos = 20
        step = int(frame_count / number_of_photos)

        # print("Number of frames: ", frame_count)
        # print("Step: ", step)
        i=0
        while(cap.isOpened()):
            # if ret = False, video finished
            ret, frame = cap.read()
            
            # (i+1) to account for the right number of photos
            if not (i + 1) % step:
                # a = cv2.waitKey(0) # close window when ESC is pressed

                # if a == 27:
                    # break               

                if ret == False:
                    cap.release()
                    cv2.destroyAllWindows()
                
                else:
                    # cv2.imshow('image', frame)
                    frame = cv2.resize(frame, targetSize, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(target_folder + str(img_number)+'.jpg', frame)
                    # print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                    img_number+=1

            i+=1

def main():
    # Set the targetSize
    target_folder='D:/1. Github/object_detection_for_grasping/images_teste_3/' # do not forget to add '/' at the end
    source_files = glob.glob("D:/1. Github/object_detection_for_grasping/images_teste_3/videos/*.mp4")
    print('Number of files: ', len(source_files))

    targetSize = (800, 800) #  width / height
    image_extractor(targetSize, target_folder, source_files)

if __name__ == "__main__":
    main()