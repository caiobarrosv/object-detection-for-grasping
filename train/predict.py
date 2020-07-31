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

class Detector:
    def __init__(self, model_path, model='ssd300', ctx='gpu', threshold=0.5, device_id=1):
        self.data_common = dataset_commons.get_dataset_files()
        self.model_path = os.path.join(self.data_common['checkpoint_folder'], model_path)
        self.threshold = threshold
        self.device_id = device_id
        
        classes_keys = [key for key in self.data_common['classes']]
        self.classes = classes_keys
        
        if ctx == 'cpu':
            self.ctx = mx.cpu()
        elif ctx == 'gpu':
            self.ctx = mx.gpu(0)
        else:
            raise ValueError('Invalid context.')
        
        if model.lower() == 'ssd300':
            model_name = 'ssd_300_vgg16_atrous_voc' #'ssd_300_vgg16_atrous_coco'
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
        image = os.path.join(self.data_common['image_folder'], image)
        # image_tensor, image = self.ssd_val_transform(image)
        image_tensor, image = gcv.data.transforms.presets.ssd.load_test(image, self.width)
        # x, image = self.transform(image, 300)
        labels, scores, bboxes = self.net(image_tensor.as_in_context(self.ctx))

        self.labels = labels
        self.scores = scores
        self.bboxes = bboxes
        self.image = image

        if plot:
            self.plot_boxes_and_image()

    def detect_webcam(self, NUM_FRAMES=200):
        device_id = self.device_id
        # Load the webcam handler
        cap = cv2.VideoCapture(device_id) # 1 for droid-cam
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
            print(fclass_IDs.size)
            gcv.utils.viz.cv_plot_image(frame)
            if fclass_IDs.size > 0:
                # Display the result
                img = gcv.utils.viz.cv_plot_bbox(frame, fbounding_boxes, fscores, fclass_IDs, class_names=self.net.classes)
                gcv.utils.viz.cv_plot_image(img)
            
            a = cv2.waitKey(1) # close window when ESC is pressed            
        
        cap.release()
        cv2.destroyAllWindows()

    def plot_boxes_and_image(self, absolute_coordinates=True, thresh=0.5):
        """
        Visualize bounding boxes and the image. This code is a modified version of the 
        original code provided by GluonCV. Please refers to GluonCV repo/website fore more info

        Argument:
            absolute_coordinates (bool): If `True`, absolute coordinates will be considered, otherwise coordinates
                are interpreted as in range(0, 1).
            thresh (float, optional, default 0.5): Display threshold if `scores` is provided. Scores with less 
                than `thresh` will be ignored in display, this is visually more elegant if you have
                a large number of bounding boxes with very small scores.
        """
        bboxes = self.bboxes # Shape (1, Boxes,  4) - 4 refers to each xmin, ymin, xmax, ymax
        scores = self.scores # Shape (1, Scores, 1)
        labels = self.labels # Shape (1, Labels, 1)
        image = self.image

        if isinstance(bboxes, mx.nd.NDArray):
            bboxes = bboxes.asnumpy()
        if isinstance(labels, mx.nd.NDArray):
            labels = labels.asnumpy()
        if isinstance(scores, mx.nd.NDArray):
            scores = scores.asnumpy()
        
        if not absolute_coordinates:
            # convert to absolute coordinates using image shape
            height = image.shape[0]
            width = image.shape[1]
            bboxes[:, (0, 2)] *= width
            bboxes[:, (1, 3)] *= height
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # OpenCV uses BGR order
        
        for i, bbox in enumerate(bboxes[0]):
            if scores is not None and scores.flat[i] < thresh:
                continue
            if labels is not None and labels.flat[i] < 0:
                continue
        
            cls_id = int(labels.flat[i]) if labels is not None else -1
            
            xmin, ymin, xmax, ymax = [int(x) for x in bbox]

            if labels is not None and cls_id < len(labels):
                class_name = self.classes[cls_id]
            else:
                class_name = str(cls_id) if cls_id >= 0 else ''
            
            score = '{:.3f}'.format(scores.flat[i]) if scores is not None else ''

            if class_name or score:
                cv2.startWindowThread()
                image = cv2.putText(image, class_name + ': ' + score, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
                image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 1)
                # h, w, _ = image.shape
                # bbox = tbbox.resize(bbox, in_size=(h, w), out_size=(self.width, self.height))
        
        cv2.imshow('image', image)
        a = cv2.waitKey(0) # close window when ESC is pressed
        cv2.destroyWindow('image')
            
def main():
    # You just need to pass folder name inside the log folder (checkpoints folder configured in config.json)
    params = 'teste_4/checkp_best_epoch_0003_map_0.9545.params'
    
    det = Detector(params, model='ssd300', ctx='gpu', threshold=0.7, device_id=1)

    # imagem = '315.jpg'    
    # det.detect(imagem, plot=True)

    number_of_frames = 200
    det.detect_webcam(number_of_frames)


if __name__ == "__main__":
    main()