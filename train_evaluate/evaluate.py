import os
import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon
# from gluoncv.data.transforms.presets import ssd, rcnn
from gluoncv.model_zoo import get_model
from gluoncv.utils import viz
from gluoncv import data as gdata
import gluoncv.data.transforms.image as timage
import gluoncv.data.transforms.bbox as tbbox
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import utils.dataset_commons as dataset_commons
import time
import glob
from matplotlib import pyplot as plt
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.utils.bbox import bbox_iou 
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform

data_common = dataset_commons.get_dataset_files()

class Detector:
    def __init__(self, model_path, model='ssd300', ctx='gpu', threshold=0.5, validation_threshold=0.5, batch_size=4, num_workers=2, nms_threshold=0.5):
        self.model_path = model_path
        self.threshold = threshold
        self.validation_threshold = validation_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nms_threshold = nms_threshold
        
        classes_keys = [key for key in data_common['classes']]
        self.classes = classes_keys
        
        if ctx == 'cpu':
            self.ctx = [mx.cpu()]
        elif ctx == 'gpu':
            self.ctx = [mx.gpu(0)]
        else:
            raise ValueError('Invalid context.')
        
        if model.lower() == 'ssd300_vgg16_voc':
            model_name = 'ssd_300_vgg16_atrous_voc' #'ssd_300_vgg16_atrous_coco'
            self.dataset= 'voc'
            self.width, self.height = 300, 300
        elif model.lower() == 'ssd300_vgg16_coco':
            model_name = 'ssd_300_vgg16_atrous_coco'
            self.dataset= 'voc' # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< MUDAR ISSO DEPOIS PARA COCO
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

        # TODO: load the train and val rec file
        self.val_file = data_common['record_val_path']

        if self.dataset == 'voc':
            self.val_dataset = gdata.RecordFileDetection(self.val_file)
            self.val_metric = VOC07MApMetric(iou_thresh=validation_threshold, class_names=self.net.classes)
        else:
            raise NotImplementedError('Dataset: {} not implemented.'.format(self.dataset))

        # Val verdadeiro
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = gluon.data.DataLoader(
            self.val_dataset.transform(SSDDefaultValTransform(self.width, self.height)),
            batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=self.num_workers)
        self.val_loader = val_loader
    
    def filter_predictions(self, bounding_boxes, scores, class_IDs):
        threshold = self.threshold
        idx = scores.squeeze().asnumpy() > threshold
        fscores = scores.squeeze().asnumpy()[idx]
        fids = class_IDs.squeeze().asnumpy()[idx]
        fbboxes = bounding_boxes.squeeze().asnumpy()[idx]
        return fbboxes, fscores, fids 

    def show_images(self, data, gt_bbox, det_bbox, index):
        # Function still unused but kept for backup
        xmin_gt, ymin_gt, xmax_gt, ymax_gt = [int(x) for x in gt_bbox[0]]
        xmin_pred, ymin_pred, xmax_pred, ymax_pred = [int(x) for x in det_bbox[0]]
        img = data[index]
        img = img.transpose((1, 2, 0))  # Move channel to the last dimension
        # img = img.asnumpy().astype('uint8') # convert to numpy array
        # img = img.astype(np.uint8)  # use uint8 (0-255)
        img = img.asnumpy()
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # OpenCV uses BGR orde
        cv2.rectangle(img, (xmin_gt, ymin_gt), (xmax_gt, ymax_gt), (255, 0, 0), 1)
        cv2.rectangle(img, (xmin_pred, ymin_pred), (xmax_pred, ymax_pred), (0, 255, 0), 1)
        cv2.startWindowThread()
        cv2.imshow('img', img)
        cv2.waitKey(5000)
        cv2.destroyWindow('img')

    def validate(self):
        """Test on validation dataset."""
        val_data = self.val_loader
        ctx = self.ctx
        val_metric = self.val_metric
        nms_threshold = self.nms_threshold
        validation_threshold = self.validation_threshold

        val_metric.reset()
        # set nms threshold and topk constraint
        # post_nms = maximum number of objects per image
        self.net.set_nms(nms_thresh=nms_threshold, nms_topk=200, post_nms=len(self.classes)) # default: iou=0.45 e topk=400

        # allow the MXNet engine to perform graph optimization for best performance.
        self.net.hybridize(static_alloc=True, static_shape=True)

        num_of_classes = len(self.classes)
        # total number of correct prediction by class
        tp = [0] * num_of_classes
        # count the number of gt by class
        gt_by_class = [0] * num_of_classes
        # false positives by class
        fp = [0] * num_of_classes
        # rec and prec by class
        rec_by_class = [0] * num_of_classes
        prec_by_class = [0] * num_of_classes
        confusion_matrix = np.zeros((num_of_classes, num_of_classes))

        for batch in val_data:
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

            pred_bboxes_list = []
            pred_label_list = []
            pred_scores_list = []
            gt_bboxes_list = []
            gt_label_list = []
            
            for x, y in zip(data, label):
                # get prediction results
                ids, scores, bboxes = self.net(x)
                pred_label_list.append(ids)
                pred_scores_list.append(scores)
                # clip to image size
                pred_bboxes_list.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_label_list.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes_list.append(y.slice_axis(axis=-1, begin=0, end=4))
            
            # update metric
            val_metric.update(pred_bboxes_list, pred_label_list, pred_scores_list, gt_bboxes_list, gt_label_list) #, gt_difficults)

            # Get Micro Averaging (precision and recall by each class) in each batch
            for img in range(batch_size):
                gt_label_ordenado, gt_bboxes_ordenado = [], []
                for ids in pred_label_list[0][img]:
                    pred_label = (int(ids.asnumpy()[0]))
                    pred_bbox = pred_bboxes_list[0][img][pred_label]
                    
                    # It is required to check if the predicted class is in the image
                    # otherwise, count it as a false positive and do not include in the list
                    if pred_label in list(gt_label_list[0][img]):    
                        # it will only order the ground truths according to the
                        # pred labels in order to compare correctly later on
                        gt_index = list(gt_label_list[0][img]).index(pred_label)
                        gt_label = gt_label_list[0][img][gt_index]
                        gt_bbox = gt_bboxes_list[0][img][gt_index]               
                        gt_label_ordenado.extend(gt_label)
                        gt_bboxes_ordenado.append(gt_bbox)
                    else:
                        fp[pred_label] += 1  # Wrong classification
                        pred_bbox_fc = pred_bbox.asnumpy()
                        pred_bbox_fc = np.expand_dims(pred_bbox_fc, axis=0)
                        # We iterate over each ground truth and check if which one matches
                        # with the predicted bounding box and store it in the confusion matrix accordingly
                        for (gt_bbox_label, gt_bbox_coordinates) in zip(gt_label_list[0][img], list(gt_bboxes_list[0][img])):
                            gt_bbox_coord = gt_bbox_coordinates.asnumpy()
                            gt_bbox_coord = np.expand_dims(gt_bbox_coord, axis=0)
                            iou_prev = bbox_iou(pred_bbox_fc, gt_bbox_coord)
                            # self.show_images(x, gt_bbox_coord, pred_bbox_fc, img)
                            if iou_prev > validation_threshold:
                                gt_bbox_label = int(gt_bbox_label.asnumpy()[0])
                                # the network though that the gt_bbox_label was the pred_label
                                # plot the image and check
                                confusion_matrix[gt_bbox_label][pred_label] += 1
                                break

                xww = 1

                # count +1 for this class id. It will get the total number of gt by class
                # It is useful when considering unbalanced datasets
                for gt_idx in gt_label_list[0][img]:
                    index = int(gt_idx.asnumpy()[0])
                    gt_by_class[index] += 1
                
                for ids in range(len(gt_bboxes_ordenado)):
                    pred_bbox_ids = pred_bboxes_list[0][img][ids]
                    pred_bbox_ids = pred_bbox_ids.asnumpy()
                    pred_bbox_ids = np.expand_dims(pred_bbox_ids, axis=0) # each bounding box related to the inference
                    predict_ind = int(pred_label_list[0][img][ids].asnumpy()[0]) # each inference label

                    gt_bbox_ids = gt_bboxes_ordenado[ids]
                    gt_bbox_ids = gt_bbox_ids.asnumpy()
                    gt_bbox_ids = np.expand_dims(gt_bbox_ids, axis=0) # each bounding box related to the ground-truth
                    gt_ind = int(gt_label_ordenado[ids].asnumpy()[0]) # each gt label
                    
                    iou = bbox_iou(pred_bbox_ids, gt_bbox_ids)

                    # Uncomment the following line if you want to plot the images in each inference to visually  check the tp, fp and fn 
                    # self.show_images(x, gt_bbox_ids, pred_bbox_ids, img)
                    
                    # Check if IoU is above the threshold and the class id corresponds to the ground truth
                    if (iou > validation_threshold) and (predict_ind == gt_ind):
                        tp[gt_ind] += 1 # Correct classification
                        confusion_matrix[gt_ind][gt_ind]  += 1
                    else:
                        fp[predict_ind] += 1  # Wrong classification
                        for (gt_bbox_label, gt_bbox_coordinates) in zip(gt_label_list[0][img], list(gt_bboxes_list[0][img])):
                            gt_bbox_coord = gt_bbox_coordinates.asnumpy()
                            gt_bbox_coord = np.expand_dims(gt_bbox_coord, axis=0)
                            iou_prev = bbox_iou(pred_bbox_ids, gt_bbox_coord)
                            self.show_images(x, gt_bbox_coord, pred_bbox_ids, img)
                            if iou_prev > validation_threshold:
                                gt_bbox_label = int(gt_bbox_label.asnumpy()[0])
                                confusion_matrix[gt_bbox_label][predict_ind] += 1
                                break         
        
        # calculate the Recall and Precision by class
        tp = np.array(tp)
        fp = np.array(fp)

        # rec and prec according to the micro averaging
        for i, (gt_value, tp_value) in enumerate(zip(gt_by_class, tp)):
            rec_by_class[i] += tp_value/gt_value

            # If an element of fp + tp is 0,
            # the corresponding element of prec[l] is nan.
            with np.errstate(divide='ignore', invalid='ignore'):
                prec_by_class[i] += tp_value/(tp_value+fp[i])

        rec, prec = val_metric._recall_prec()
        gt_by_class_sum = sum(gt_by_class)
        fp_sum = sum(fp)
        tp_sum = sum(tp)
        return val_metric.get(), rec_by_class, prec_by_class, gt_by_class_sum, fp_sum, tp_sum

    def detect(self, image, plot=False):
        image_tensor, image = gcv.data.transforms.presets.ssd.load_test(image, self.width)
        labels, scores, bboxes = self.net(image_tensor.as_in_context(self.ctx))
        if plot:
            ax = viz.plot_bbox(image, bboxes[0], scores[0], labels[0], class_names=self.net.classes)
            plt.show()

def evaluation_analysis(experiments_ids_list, fp_sum_list, tp_sum_list, prec_by_class_list):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Network evaluation')
    red_square = dict(markerfacecolor='r', marker='s')
    ax1.set_title('Precision analysis')
    ax1.boxplot(prec_by_class_list, labels=experiments_ids_list, flierprops=red_square)
    ax2.set_ylabel('Precision')
    ax1.set_xlabel('Experiment ID')
    width = 0.35
    x = np.arange(len(experiments_ids_list))
    rects1 = ax2.bar(x - width/2, fp_sum_list, width, label='False Positives')
    rects2 = ax2.bar(x + width/2, tp_sum_list, width, label='True Positives')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax2.set_ylabel('Detections')
    ax2.set_title('FP/TP')
    ax2.set_xticks(x)
    ax2.set_xticklabels(experiments_ids_list)
    ax2.legend(loc='center left')
    ax2.set_xlabel('Experiment ID')
    #def autolabel(rects):
    #   """Attach a text label above each bar in *rects*, displaying its height."""
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax2.annotate('{}'.format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom')
    fig.tight_layout()
    plt.show()

def main():
    model_networks_path = glob.glob(data_common['checkpoint_folder'] + '/*/')

    experiments_ids_list = []
    prec_by_class_list = []
    model_names_list = []
    gt_by_class_sum_list = []
    fp_sum_list = []
    tp_sum_list = []
    for model_path in model_networks_path:
        experiment_paths = glob.glob(model_path + '/*/')
        start = model_path.find('ints')
        model_name = model_path[start+5:-1]
        for experiment_path in experiment_paths:
            param_paths = glob.glob(experiment_path + '/*.params')
            start = experiment_path.find('SAN')
            experiment_id_name = experiment_path[start:-1]
            for param_path in param_paths:
                start = param_path.find(experiment_id_name)
                param_name = param_path[start+len(experiment_id_name)+2:-1]

                det = Detector(param_path, 
                   model=model_name, 
                   ctx='gpu', 
                   threshold=0.8, # Used to filter the bounding boxes after detection - it will not affect validation 
                   batch_size=4, 
                   num_workers=2, 
                   nms_threshold=0.5 # It will affect validation
                   )
                (map_name, mean_ap), rec_by_class, prec_by_class, gt_by_class_sum, fp_sum, tp_sum = det.validate()
                map_geral = float(mean_ap[-1])
                map_gera_name = map_name[-1]

                prec_by_class_list.append(prec_by_class)
                model_names_list.append(model_name)
                experiments_ids_list.append(experiment_id_name)
                gt_by_class_sum_list.append(gt_by_class_sum)
                fp_sum_list.append(fp_sum)
                tp_sum_list.append(tp_sum)

                print('Model_Name: ', model_name)
                print('Experiment_id: ', experiment_id_name)
                print('params: ', param_name)
                print('pec: ', prec_by_class)

    
    evaluation_analysis(experiments_ids_list, fp_sum_list, tp_sum_list, prec_by_class_list)
    xsdq = 1
    # os.path.join(data_common['checkpoint_folder'], self.model_name)

    # # TODO: You just need to pass the param name inside the log folder (checkpoints folder configured in config.json)
    # params = 'ssd_300_vgg16_atrous_voc_best_epoch_0025_map_0.8749.params'

    # 
    
    # (map_name, mean_ap), rec_by_class, prec_by_class = det.validate()
    # map_geral = float(mean_ap[-1])
    # map_gera_name = map_name[-1]

if __name__ == "__main__":
    main()