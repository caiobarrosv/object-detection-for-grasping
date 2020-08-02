import os
import mxnet as mx
import gluoncv as gcv
# from gluoncv.data.transforms.presets import ssd, rcnn
from gluoncv.model_zoo import get_model
from gluoncv.utils import viz
import gluoncv.data.transforms.image as timage
import gluoncv.data.transforms.bbox as tbbox
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import utils.dataset_commons as dataset_commons
import time
import glob
from matplotlib import pyplot as plt

data_common = dataset_commons.get_dataset_files()

class Detector:
    def __init__(self, model_path, model='ssd300', ctx='gpu', threshold=0.5, device_id=1):
        self.model_path = os.path.join(data_common['checkpoint_folder'], model_path)
        self.threshold = threshold
        self.device_id = device_id
        
        classes_keys = [key for key in data_common['classes']]
        self.classes = classes_keys
        
        if ctx == 'cpu':
            self.ctx = mx.cpu()
        elif ctx == 'gpu':
            self.ctx = mx.gpu(0)
        else:
            raise ValueError('Invalid context.')
        
        if model.lower() == 'ssd300_vgg16_voc':
            model_name = 'ssd_300_vgg16_atrous_voc' #'ssd_300_vgg16_atrous_coco'
            self.width, self.height = 300, 300
        elif model.lower() == 'ssd300_vgg16_coco':
            model_name = 'ssd_300_vgg16_atrous_coco'
            self.width, self.height = 300, 300
        elif (model.lower() == 'frcnn'):
            model_name = 'faster_rcnn_resnet50_v1b_coco'
            short = 600 # used to transform the images for the Faster R-CNN
            # ongoing
            # self.transform = rcnn.FasterRCNNDefaultValTransform(short=short)
        else:
            raise ValueError('Invalid model `{}`.'.format(model.lower()))
        
        net = get_model(model_name, pretrained=False, ctx=self.ctx)
        # net.set_nms(nms_thresh=0.5, nms_topk=2)
        net.hybridize(static_alloc=True, static_shape=True)
        net.initialize(force_reinit=True, ctx=self.ctx)
        print(self.classes)
        net.reset_class(classes=self.classes)
        net.load_parameters(self.model_path, ctx=self.ctx)
		
        self.net = net
    
    def filter_predictions(self, bounding_boxes, scores, class_IDs):
        threshold = self.threshold
        idx = scores.squeeze().asnumpy() > threshold
        fscores = scores.squeeze().asnumpy()[idx]
        fids = class_IDs.squeeze().asnumpy()[idx]
        fbboxes = bounding_boxes.squeeze().asnumpy()[idx]
        return fbboxes, fscores, fids 

    def ssd_val_transform(self, img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        if isinstance(img, str):
            img = [img]
        imgs = [mx.image.imread(f) for f in img]

        if isinstance(imgs, mx.nd.NDArray):
            imgs = [imgs]
        for im in imgs:
            assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

        tensors = []
        origs = []
        for img in imgs:
            img = timage.resize_short_within(img, self.width, max_size=1024)
            orig_img = img.asnumpy().astype('uint8')
            img = mx.nd.image.to_tensor(img)
            img = mx.nd.image.normalize(img, mean=mean, std=std)
            tensors.append(img.expand_dims(0))
            origs.append(orig_img)
        if len(tensors) == 1:
            return tensors[0], origs[0]
        return tensors, origs
        
    def detect(self, image, plot=False):
        # image_tensor, image = self.ssd_val_transform(image)
        image_tensor, image = gcv.data.transforms.presets.ssd.load_test(image, self.width)
        # x, image = self.transform(image, 300)
        labels, scores, bboxes = self.net(image_tensor.as_in_context(self.ctx))
        if plot:
            ax = viz.plot_bbox(image, bboxes[0], scores[0], labels[0], class_names=self.net.classes)
            plt.show()

    def detect_webcam_video(self, video_font):
        # Load the webcam handler
        cap = cv2.VideoCapture(video_font) # 1 for droid-cam
        time.sleep(1) ### letting the camera autofocus

        axes = None
        a = cv2.waitKey(0) # close window when ESC is pressed     
        while a is not 27:
            # Load frame from the camera
            ret, frame = cap.read()

            # Image pre-processing
            frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
            rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=self.width, max_size=700)

            # Run frame through network
            class_IDs, scores, bounding_boxes = self.net(rgb_nd.as_in_context(self.ctx))          

            fbounding_boxes, fscores, fclass_IDs = self.filter_predictions(bounding_boxes, scores, class_IDs)
            gcv.utils.viz.cv_plot_image(frame)
            if fclass_IDs.size > 0:
                # Display the result
                img = gcv.utils.viz.cv_plot_bbox(frame, fbounding_boxes, fscores, fclass_IDs, class_names=self.net.classes)
                gcv.utils.viz.cv_plot_image(img)
            
            a = cv2.waitKey(1) # close window when ESC is pressed            
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    # TODO: You just need to pass the param name inside the log folder (checkpoints folder configured in config.json)
    params = 'ssd_300_vgg16_coco_dataset_5_epoch_0059_map_1.0000_loss_1.0.params'
    
    det = Detector(params, model='ssd300_vgg16_coco', ctx='gpu', threshold=0.1, device_id=1)

    print("\nPlease configure the video/images files path in config.json before running the next command.")    
    a = int(input("Choose to inferring by using: \n[option: 1] - Images \n[option: 2] - Videos \n[option: 3] - Webcam\nOption: "))
    
    if a == 1:
        images = glob.glob(data_common['image_folder'] + "/" + "*.jpg")
        for image in images:
            det.detect(image, plot=True)
    elif a == 2:
        file_name = str(input("Write the video file name with the extension (!!) that is inside the video folder configured in the config.json file: "))
        file_name = glob.glob(data_common['video_folder'] + "/" + file_name)[0]
        det.detect_webcam_video(file_name)
    elif a == 3:
        device_id = int(input("Choose the device id to connect (default: 0): "))
        det.detect_webcam_video(device_id)
    else:
        print("Please choose the right option")

if __name__ == "__main__":
    main()