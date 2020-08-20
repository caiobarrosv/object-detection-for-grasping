"""Train SSD"""
# import argparse
import os
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
# from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.bbox import bbox_iou 
import cv2
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import utils.common as dataset_commons
import pandas as pd

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

'''
ATTENTION!

This script will generate a csv file in the folder 'checkpoints' configured in the config.json
this csv file will contain the following data:
['model_name', 'map_iou_0.5', 'map_iou_0.75', 'map_iou_0.5:0.95', 'experiment_id']

Instructions:

Put all your trained network params inside the checkpoints folder in the following way:
    checkpoints/network_name/experiment_id/params

Example:
    checkpoints/ssd_300_vgg16_atrous_voc/SAN-8/ssd_300_vgg16_atrous_voc_best_epoch_0042_map_0.9409.params

The models used in this project:
    PASCAL VOC:
        1) ssd_300_vgg16_atrous_voc        
        2) ssd_512_vgg16_atrous_voc 
        3) ssd_512_ResNet50_atrous_voc
        4) faster_rcnn_resnet50_v1b_voc
        5) yolo3_darknet53_voc 
    COCO:
        1) ssd_300_vgg16_atrous_coco 
        2) ssd_512_vgg16_atrous_coco 
        3) ssd_512_resnet50_v1_coco 
        4) faster_rcnn_resnet50_v1b_coco 
        5) yolo3_darknet53_coco 
'''

data_common = dataset_commons.get_dataset_files()

class training_network():
    def __init__(self, model='ssd300', ctx='gpu', batch_size=4, num_workers=2, 
                 validation_threshold=0.5, nms_threshold=0.5, param_path=None):
        """
        Script responsible for training the class

        Arguments:
            model (str): One of the following models [ssd_300_vgg16_atrous_voc]
            num_worker (int, default: 2): number to accelerate data loading
            dataset (str, default:'voc'): Training dataset. Now support voc.
            batch_size (int, default: 4): Training mini-batch size
        """

        # EVALUATION PARAMETERS
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_threshold = validation_threshold
        self.nms_threshold = nms_threshold

        if ctx == 'cpu':
            self.ctx = [mx.cpu()]
        elif ctx == 'gpu':
            self.ctx = [mx.gpu(0)]
        else:
            raise ValueError('Invalid context.')
            
        # fix seed for mxnet, numpy and python builtin random generator.
        gutils.random.seed(233)

        self.width, self.height, _ = dataset_commons.get_model_prop(model)
        self.model_name = model
        
        # TODO: load the train and val rec file
        self.val_file = data_common['record_val_path']

        self.classes = ['bar_clamp', 'gear_box', 'vase', 'part_1', 'part_3', 'nozzle', 'pawn', 'turbine_housing'] # please, follow the order of the config.json file
        print('Classes: ', self.classes)

        net = get_model(self.model_name, pretrained=False, ctx=self.ctx)
        # net.set_nms(nms_thresh=0.5, nms_topk=2)
        net.hybridize(static_alloc=True, static_shape=True)
        net.initialize(force_reinit=True, ctx=self.ctx)
        net.reset_class(classes=self.classes)
        net.load_parameters(param_path, ctx=self.ctx)
        self.net = net
    
        val_dataset = gdata.RecordFileDetection(self.val_file)

        # Val verdadeiro
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = gluon.data.DataLoader(
            val_dataset.transform(SSDDefaultValTransform(self.width, self.height)),
            batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
        
        self.val_loader = val_loader

    def update_iou(self, validation_threshold):
        self.validation_threshold = validation_threshold
        self.val_metric = VOC07MApMetric(iou_thresh=validation_threshold, class_names=self.net.classes)

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
        # false positives by class
        fp = [0] * num_of_classes
        # count the number of gt by class
        gt_by_class = [0] * num_of_classes
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

            # Uncomment the following line if you want to plot the images in each inference to visually  check the tp, fp and fn 
            # self.show_images(x, pred_label_list, pred_bboxes_list, gt_label_list, gt_bboxes_list)
            
            # update metric
            val_metric.update(pred_bboxes_list, pred_label_list, pred_scores_list, gt_bboxes_list, gt_label_list) #, gt_difficults)
            
            # Get Micro Averaging (precision and recall by each class) in each batch
            for img in range(batch_size):
                # count +1 for this class id. It will get the total number of gt by class
                # It is useful when considering unbalanced datasets
                for gt_idx in gt_label_list[0][img]:
                    index = int(gt_idx.asnumpy()[0])
                    gt_by_class[index] += 1
            
                for (pred_label, pred_bbox) in zip(pred_label_list[0][img], list(pred_bboxes_list[0][img])):
                    pred_label = int(pred_label.asnumpy()[0])
                    pred_bbox = pred_bbox.asnumpy()
                    pred_bbox = np.expand_dims(pred_bbox, axis=0)
                    match = 0
                    for (gt_bbox_label, gt_bbox_coordinates) in zip(gt_label_list[0][img], list(gt_bboxes_list[0][img])):
                        gt_bbox_coord = gt_bbox_coordinates.asnumpy()
                        gt_bbox_coord = np.expand_dims(gt_bbox_coord, axis=0)
                        gt_bbox_label = int(gt_bbox_label.asnumpy()[0])
                        iou = bbox_iou(pred_bbox, gt_bbox_coord)
                        
                        # Correct inference
                        if iou > validation_threshold and pred_label == gt_bbox_label:
                            confusion_matrix[gt_bbox_label][pred_label] += 1
                            tp[gt_bbox_label] += 1 # Correct classification
                            match = 1
                        # Incorrect inference - missed the correct class but put the bounding box in other class
                        elif iou > validation_threshold:
                            confusion_matrix[gt_bbox_label][pred_label] += 1
                            fp[pred_label] += 1
                            match = 1
                        
                    if not match:
                        fp[pred_label] += 1
                                
        # calculate the Recall and Precision by class
        tp = np.array(tp) # we can also sum the matrix diagonal
        fp = np.array(fp)
        
        fp_sum = sum(fp)
        tp_sum = sum(tp)

        # rec and prec according to the micro averaging
        for i, (gt_value, tp_value) in enumerate(zip(gt_by_class, tp)):
            rec_by_class[i] += tp_value/gt_value

            # If an element of fp + tp is 0,
            # the corresponding element of prec[l] is nan.
            with np.errstate(divide='ignore', invalid='ignore'):
                prec_by_class[i] += tp_value/(tp_value+fp[i])
        # rec, prec = val_metric._recall_prec()      
        return val_metric.get(), rec_by_class, prec_by_class, fp_sum, tp_sum

    def evaluate_main(self):
        """Training pipeline"""
        val_data = self.val_loader
        eval_metric = self.val_metric
        ctx = self.ctx
        
        print('Analyzing validation threshold: [{}] ...'.format(self.validation_threshold))
        
        (map_name, mean_ap), rec_by_class, prec_by_class, fp_sum, tp_sum = self.validate()
        val_msg = '\n'.join(['{}={} | prec: [{}]'.format(k, v, x) for k, v, x in zip(map_name, mean_ap, rec_by_class)])
        print(val_msg)      
        best_map = mean_ap[-1]
        return best_map, mean_ap, prec_by_class, fp_sum, tp_sum

if __name__ == '__main__':
    threshold = [0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.9, 0.95]
    
    coco_metric_dic_list = []

    model_networks_path = glob.glob(data_common['checkpoint_folder'] + '/*/')
    csv_path_save = data_common['checkpoint_folder'] + '/coco_evaluation.csv'
    column_name = ['model_name', 'map_iou_0.5', 'map_iou_0.75', 'map_iou_0.5:0.95', 'experiment_id']
    csv_list = []
    
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
                train_object = training_network(model=model_name, 
                                                    ctx='gpu',
                                                    batch_size=4,
                                                    param_path=param_path)
                start_train_time = time.time()
                best_map_list = []
                mean_ap_voc = []
                prec_list = []
                for thresh in threshold: 
                    train_object.update_iou(thresh)
                    best_map, mean_ap, prec_by_class, fp_sum, tp_sum = train_object.evaluate_main()
                    # We just want to analyze the mAPs per object using IoU of 0.5
                    if thresh == 0.5:
                        print('Saving mAPs per object...')
                        mean_ap_voc = mean_ap
                        prec_list.append(prec_by_class) # precision is only analyzed using IoU = 0.5
                        fp_sum_iou_05 = fp_sum
                        tp_sum_iou_05 = tp_sum
                    print('best map: [{}] | threshold: [{}] \n'.format(best_map, thresh))
                    best_map_list.append(best_map)
                
                map_05 = round(best_map_list[0]*100, 1)
                map_075 = round(best_map_list[5]*100, 1)
                media_05_095 = round(sum(best_map_list)*100/len(best_map_list), 1)
                coco_metric_dic_list.append({'model_name': model_name, 
                                             'map_iou_0.5': map_05, 
                                             'map_iou_0.75' : map_075, 
                                             'map_iou_0.5:0.95' : media_05_095,
                                             'experiment_id' : experiment_id_name})
                
                bar_clamp_map, gear_box_map, vase_map, part_1_map, part_3_map, \
                    nozzle_map, pawn_map, turbine_housing_map, map_ = [round(map_*100, 2) for map_ in mean_ap_voc]
                
                bar_clamp_prec, gear_box_prec, vase_prec, part_1_prec, part_3_prec, \
                    nozzle_prec, pawn_prec, turbine_housing_prec = [round(prec*100,2) for prec in prec_list[0]]

                value = (model_name,
                         map_05, map_075, media_05_095,
                         experiment_id_name,
                         bar_clamp_map, gear_box_map, vase_map,
                         part_1_map, part_3_map, nozzle_map,
                         pawn_map, turbine_housing_map, fp_sum_iou_05, tp_sum_iou_05,
                         bar_clamp_prec, gear_box_prec, vase_prec, part_1_prec, part_3_prec,
                         nozzle_prec, pawn_prec, turbine_housing_prec
                         )   
                csv_list.append(value)

                print('{} - mAPs [0.5:0.05:0.95]: {}'.format(model_name, best_map_list))
                print('{} - Evaluation time [min]: {:.3f}'.format(model_name, (time.time() - start_train_time)/60))
                print('{} - mAP IoU 0.5: [{}] | mAP  IoU 0.75: [{}] | mAP  IoU 0.5:0.95: [{}]'.format(model_name, map_05, map_075, media_05_095))
                print('{} - Precision by class with IoU 0.5: {} \n'.format(model_name, prec_list))
    
    column_name = ['model_name', 'map_iou_0.5', 'map_iou_0.75', 'map_iou_0.5:0.95', 'experiment_id', 'bar_clamp_map_iou_0.5',
                   'gear_box_map_iou_0.5', 'vase_map_iou_0.5', 'part_1_map_iou_0.5', 'part_3_map_iou_0.5',
                   'nozzle_map_iou_0.5', 'pawn_map_iou_0.5', 'turbine_housing_map_iou_0.5', 'false_positives', 'true_positives',
                   'bar_clamp_prec_iou_0.5', 'gear_box_prec_iou_0.5', 'vase_prec_iou_0.5', 'part_1_prec_iou_0.5', 'part_3_prec_iou_0.5',
                   'nozzle_prec_iou_0.5', 'pawn_prec_iou_0.5', 'turbine_housing_prec_iou_0.5',]
    csv_df = pd.DataFrame(csv_list, columns=column_name)
    csv_df.to_csv(csv_path_save, index=None)
    print(coco_metric_dic_list)
    print('csv saved in: ', csv_path_save)