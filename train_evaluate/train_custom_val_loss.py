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
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
# from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.bbox import bbox_iou 
from mxnet.contrib import amp
import cv2

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import utils.common as dataset_commons
from utils.ssd_custom_val_transform import SSDCustomValTransform
import utils.environments_setup # must be imported before NEPTUNE
import neptune

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'

'''
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
    def __init__(self, model='ssd300', ctx='gpu', resume_training=False, batch_size=4, num_workers=2, lr=0.001, 
                 lr_decay=0.1, lr_decay_epoch='60, 80', wd=0.0005, momentum=0.9, start_epoch=0,
                 epochs=2, dataset='voc', network='vgg16_atrous', resume='',
                 beta1=0.9, beta2=0.999, epsilon=1e-08, validation_threshold=0.5, nms_threshold=0.5, optimizer='sgd', 
                 exp=False):
        """
        Script responsible for training the class

        Arguments:
            wd (float, default: 0.0005): Weight decay, default is 5e-4
            momentum (float, default:0.9): SGD momentum, default is 0.9
            lr_decay_epoch (str, default: '60, 80'): epoches at which learning rate decays. default is 60, 80.
            lr_decay (float, default: 0.1): decay rate of learning rate. default is 0.1.
            lr (float, default: 0.001): Learning rate, default is 0.001
            start_epoch (int, default: 0): Starting epoch for resuming, default is 0 for new training. You can
                specify it to 100 for example to start from 100 epoch.
            resume (str, default: ''): Resume from previously saved parameters if not None. For example, you 
                can resume from ./ssd_xxx_0123.params'
            epochs (int, default:2): Training epochs.
            num_worker (int, default: 2): number to accelerate data loading
            dataset (str, default:'voc'): Training dataset. Now support voc.
            batch_size (int, default: 4): Training mini-batch size
            data_shape (int, default: 300): Input data shape, use 300, 512.
            network (str, default:'vgg16_atrous'): Base network name which serves as feature extraction base.
        """

        # amp.init()

        # TRAINING PARAMETERS
        self.resume_training = resume_training
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.learning_rate = lr
        self.weight_decay = wd
        self.momentum = momentum
        self.optimizer = optimizer
        self.lr_decay = lr_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.resume = resume
        self.validation_threshold = validation_threshold
        self.nms_threshold = nms_threshold
        self.experiment = experiment
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon
        self.best_map = 0

        if ctx == 'cpu':
            self.ctx = [mx.cpu()]
        elif ctx == 'gpu':
            self.ctx = [mx.gpu(0)]
        else:
            raise ValueError('Invalid context.')
            
        # fix seed for mxnet, numpy and python builtin random generator.
        gutils.random.seed(233)

        self.width, self.height, self.network = dataset_commons.get_model_prop(model)
                
        # TODO: Specify the checkpoints save path
        self.save_prefix = os.path.join(data_common['checkpoint_folder'], model)

        # TODO: load the train and val rec file
        self.train_file = data_common['record_train_path']
        self.val_file = data_common['record_val_path']

        self.classes = ['bar_clamp', 'gear_box', 'vase', 'part_1', 'part_3', 'nozzle', 'pawn', 'turbine_housing'] # please, follow the order of the config.json file
        print('Classes: ', self.classes)

        # pretrained or pretrained_base?
        # pretrained (bool or str) – Boolean value controls whether to load the default 
        # pretrained weights for model. String value represents the hashtag for a certain 
        # version of pretrained weights.
        # pretrained_base (bool or str, optional, default is True) – Load pretrained base 
        # network, the extra layers are randomized. Note that if pretrained is True, this
        # has no effect.
        self.net = get_model(model, pretrained=True, norm_layer=gluon.nn.BatchNorm)
        self.net.reset_class(self.classes)
        
        # Initialize the weights
        if self.resume_training:
            self.net.initialize(force_reinit=True, ctx=self.ctx)
            self.net.load_params(self.resume, ctx=self.ctx)
        else:
            for param in self.net.collect_params().values():
                if param._data is not None:
                    continue
                param.initialize()
        print('aqui')

    def get_dataset(self):
        validation_threshold = self.validation_threshold
        self.train_dataset = gdata.RecordFileDetection(self.train_file)
        self.val_dataset = gdata.RecordFileDetection(self.val_file)
        # we are only using VOCMetric for evaluation
        self.val_metric = VOC07MApMetric(iou_thresh=validation_threshold, class_names=self.net.classes)

    def show_summary(self):
        self.net.summary(mx.nd.ones((1, 3, self.height, self.width)))

    def get_dataloader(self):
        width, height = self.width, self.height
        train_dataset = self.train_dataset
        val_dataset = self.val_dataset
        batch_size = self.batch_size
        num_workers = self.num_workers
        network = self.network
        print('aqui 0')
        if network == 'ssd':
            # use fake data to generate fixed anchors for target generation
            with autograd.train_mode():
                _, _, anchors = self.net(mx.nd.zeros((1, 3, height, width)))

            batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
            train_loader = gluon.data.DataLoader(train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
                                                 batch_size, True, 
                                                 batchify_fn=batchify_fn, 
                                                 last_batch='rollover', 
                                                 num_workers=num_workers)

            # Val verdadeiro
            val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
            val_loader = gluon.data.DataLoader(val_dataset.transform(SSDDefaultValTransform(width, height)),
                                               batch_size, False, 
                                               batchify_fn=val_batchify_fn, 
                                               last_batch='keep', 
                                               num_workers=num_workers)
          
            # use fake data to generate fixed anchors for target generation
            with mx.Context(mx.gpu(0)):
                anchors2 = anchors

            val_loader_loss = gluon.data.DataLoader(val_dataset.transform(SSDCustomValTransform(width, height, anchors2)),
                                                    batch_size, True, 
                                                    batchify_fn=batchify_fn, 
                                                    last_batch='rollover', 
                                                    num_workers=num_workers)
            self.val_loader_loss = val_loader_loss
        elif network == 'yolo':
            print('aqui 1')
            batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))  # stack image, all targets generated
            # if args.no_random_shape:
            train_loader = gluon.data.DataLoader(train_dataset.transform(YOLO3DefaultTrainTransform(width, height, self.net)),
                                                 batch_size, True, 
                                                 batchify_fn=batchify_fn, 
                                                 last_batch='rollover', 
                                                 num_workers=num_workers)

            val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
            val_loader = gluon.data.DataLoader(val_dataset.transform(YOLO3DefaultValTransform(width, height)),
                                               batch_size, False, 
                                               batchify_fn=val_batchify_fn, 
                                               last_batch='keep', 
                                               num_workers=num_workers)
            print('aqui 2')
        else:
            raise ValueError("Network {} not implemented".format(network))

        self.val_loader = val_loader
        self.train_loader = train_loader

    def save_params(self, current_map, epoch):
        prefix = self.save_prefix
        best_map = self.best_map

        current_map = float(current_map)        
        if current_map > best_map:
            best_map = current_map
            self.net.save_parameters('{:s}_best_epoch_{:04d}_map_{:.4f}.params'.format(prefix, epoch, current_map))
        
        self.best_map = best_map
        print('Best map: ', self.best_map)

    def val_loss(self):
        """Training pipeline"""        
        val_data = self.val_loader_loss
        ctx = self.ctx
        val_metric = self.val_metric

        mbox_loss = gcv.loss.SSDMultiBoxLoss()
        ce_metric = mx.metric.Loss('CrossEntropy')
        smoothl1_metric = mx.metric.Loss('SmoothL1')

        ce_metric.reset() # Resets the internal evaluation result to initial state.
        smoothl1_metric.reset() # Resets the internal evaluation result to initial state.

        for i, batch in enumerate(val_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)

            cls_preds = []
            box_preds = []
            for x in data:
                cls_pred, box_pred, _ = self.net(x)
                cls_preds.append(cls_pred)
                box_preds.append(box_pred)
                # descobrir o id de cada ifnerência pra usar no iou
            
            sum_loss, cls_loss, box_loss = mbox_loss(
                cls_preds, box_preds, cls_targets, box_targets)
        
        ce_metric.update(0, [l * batch_size for l in cls_loss])
        smoothl1_metric.update(0, [l * batch_size for l in box_loss])

        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()

        return name1, loss1, name2, loss2

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

        # >>>> Verificar eficácia
        # mx.nd.waitall()

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

        return val_metric.get(), rec_by_class, prec_by_class

    def create_optimizer(self):
        optimizer = self.optimizer
        momentum = self.momentum
        wd = self.weight_decay
        lr = self.learning_rate
        beta1 = self.beta1
        beta2 = self.beta2
        epsilon = self.epsilon
        
        if optimizer.lower() == 'sgd':
            # wd: The weight decay (or L2 regularization) coefficient.
            self.trainer = gluon.Trainer(self.net.collect_params(), optimizer,
                                    {'learning_rate': lr, 'wd': wd, 'momentum': momentum})
        elif optimizer.lower() == 'adam':
            self.trainer = gluon.Trainer(self.net.collect_params(), optimizer,
                                    {'learning_rate': lr, 'beta1': beta1, 'beta2': beta2, 
                                     'epsilon': epsilon})

    def validate_main(self, epoch):
        # consider reduce the frequency of validation to save time
        (map_name, mean_ap), rec_by_class, prec_by_class = self.validate()
        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        
        for i, class_name in enumerate(self.classes):
            experiment.log_metric('rec_by_class_val_' + class_name, epoch, rec_by_class[i])
            experiment.log_metric('prec_by_class_val_' + class_name, epoch, prec_by_class[i])
        
        for k, v in zip(map_name, mean_ap):
            experiment.log_metric('map_' + k, epoch, v)
        
        print('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
        current_map = float(mean_ap[-1])
        
        self.save_params(current_map, epoch)    

    def ssd_train(self):
        """Training pipeline"""
        ctx = self.ctx
        train_data = self.train_loader
        start_epoch = self.start_epoch
        epochs = self.epochs
        experiment = self.experiment
        trainer = self.trainer
        optimizer = self.optimizer

        mbox_loss = gcv.loss.SSDMultiBoxLoss()
        ce_metric = mx.metric.Loss('CrossEntropy')
        smoothl1_metric = mx.metric.Loss('SmoothL1')

        if optimizer =='sgd':
            # lr decay policy
            lr_decay = float(lr_decay)
            lr_steps = sorted([float(ls) for ls in lr_decay_epoch.split(',') if ls.strip()]) 

        print('Start training from [Epoch {}]'.format(start_epoch))
        start_train_time = time.time()
        for epoch in range(start_epoch, epochs):
            start_epoch_time = time.time()
            experiment.log_metric('learning_rate', epoch, trainer.learning_rate)

            if optimizer == 'sgd':
              while lr_steps and epoch >= lr_steps[0]:
                  new_lr = trainer.learning_rate * lr_decay
                  lr_steps.pop(0) # removes the first element in the list
                  trainer.set_learning_rate(new_lr) # Set a new learning rate
                  print("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
                
            ce_metric.reset() # Resets the internal evaluation result to initial state.
            smoothl1_metric.reset() # Resets the internal evaluation result to initial state.

            tic = time.time() # each epoch time in seconds
            btic = time.time() # each batch time interval in seconds
                
            # Activates or deactivates HybridBlocks recursively. it speeds up the training process
            self.net.hybridize(static_alloc=True, static_shape=True)
                
            for i, batch in enumerate(train_data):
                # Wait for completion of previous iteration to
                # avoid unnecessary memory allocation
                # nd.waitall()

                batch_size = batch[0].shape[0]
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
                box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)

                with autograd.record():
                    cls_preds = []
                    box_preds = []
                    for x in data:
                        cls_pred, box_pred, _ = self.net(x)
                        cls_preds.append(cls_pred)
                        box_preds.append(box_pred)
                        
                    sum_loss, cls_loss, box_loss = mbox_loss(
                        cls_preds, box_preds, cls_targets, box_targets)
                        
                    # with amp.scale_loss(sum_loss, trainer) as scaled_loss:
                        # autograd.backward(scaled_loss)
                    autograd.backward(sum_loss)
                    
                # since we have already normalized the loss, we don't want to normalize
                # by batch-size anymore
                trainer.step(1)
                ce_metric.update(0, [l * batch_size for l in cls_loss])
                smoothl1_metric.update(0, [l * batch_size for l in box_loss])

                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                print('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
                btic = time.time()

            # log the epoch info
            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            experiment.log_metric('cross_entropy_training_loss', epoch, loss1)
            experiment.log_metric('smooth_l1_training_loss', epoch, loss2) 
            experiment.log_metric('train_sum_loss', epoch, loss1 + loss2) 

            print('[Epoch {}] - Time (min): {:.3f}, {}={:.3f}, {}={:.3f}'.format(
                epoch, (time.time()-tic)/60, name1, loss1, name2, loss2))

            # log SSD LOSS            
            val_name1, val_loss1, val_name2, val_loss2 = self.val_loss()
            current_val_loss = val_loss1 + val_loss2
            experiment.log_metric('cross_entropy_validation_loss', epoch, val_loss1)
            experiment.log_metric('smooth_l1_validation_loss', epoch, val_loss2) 
            experiment.log_metric('validation_sum_loss', epoch, current_val_loss) 

            self.validate_main(epoch)
        
        # Displays the total time of the training
        print('Train time {:.3f}'.format(time.time() - start_train_time))
    
    def yolo_train(self):
        """Training pipeline"""
        ctx = self.ctx
        train_data = self.train_loader
        start_epoch = self.start_epoch
        epochs = self.epochs
        experiment = self.experiment
        trainer = self.trainer
        optimizer = self.optimizer

        # metrics
        obj_metrics = mx.metric.Loss('ObjLoss')
        center_metrics = mx.metric.Loss('BoxCenterLoss')
        scale_metrics = mx.metric.Loss('BoxScaleLoss')
        cls_metrics = mx.metric.Loss('ClassLoss')

        if optimizer =='sgd':
            # lr decay policy
            lr_decay = float(lr_decay)
            lr_steps = sorted([float(ls) for ls in lr_decay_epoch.split(',') if ls.strip()]) 

        print('Start training from [Epoch {}]'.format(start_epoch))
        start_train_time = time.time()
        for epoch in range(start_epoch, epochs):
            experiment.log_metric('learning_rate', epoch, trainer.learning_rate)

            start_epoch_time = time.time()
            tic = time.time() # each epoch time in seconds
            btic = time.time() # each batch time interval in seconds
            mx.nd.waitall()
            self.net.hybridize()
            for i, batch in enumerate(train_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                # objectness, center_targets, scale_targets, weights, class_targets
                fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 6)]
                gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
                sum_loss = []
                obj_losses = []
                center_losses = []
                scale_losses = []
                cls_losses = []
                with autograd.record():
                    for ix, x in enumerate(data):
                        obj_loss, center_loss, scale_loss, cls_loss = self.net(x, gt_boxes[ix], *[ft[ix] for ft in fixed_targets])
                        sum_loss.append(obj_loss + center_loss + scale_loss + cls_loss)
                        obj_losses.append(obj_loss)
                        center_losses.append(center_loss)
                        scale_losses.append(scale_loss)
                        cls_losses.append(cls_loss)
                    # if args.amp:
                    # with amp.scale_loss(sum_loss, trainer) as scaled_loss:
                    #         autograd.backward(scaled_loss)
                    # else:
                    autograd.backward(sum_loss)
                trainer.step(self.batch_size)
                # if (not args.horovod or hvd.rank() == 0):
                obj_metrics.update(0, obj_losses)
                center_metrics.update(0, center_losses)
                scale_metrics.update(0, scale_losses)
                cls_metrics.update(0, cls_losses)
                
                name1, loss1 = obj_metrics.get()
                name2, loss2 = center_metrics.get()
                name3, loss3 = scale_metrics.get()
                name4, loss4 = cls_metrics.get()
                print('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, trainer.learning_rate, self.batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
                btic = time.time()

            # if (not args.horovod or hvd.rank() == 0):
            name1, loss1 = obj_metrics.get()
            name2, loss2 = center_metrics.get()
            name3, loss3 = scale_metrics.get()
            name4, loss4 = cls_metrics.get()
            experiment.log_metric('Obj_metrics', epoch, loss1)
            experiment.log_metric('Center_metrics', epoch, loss2)
            experiment.log_metric('Scale_metrics', epoch, loss3)
            experiment.log_metric('cls_metrics', epoch, loss4)
            print('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))

            self.validate_main(epoch)
        
        # Displays the total time of the training
        print('Train time {:.3f}'.format(time.time() - start_train_time))
            
    def train(self):
        lr_decay = self.lr_decay
        lr_decay_epoch = self.lr_decay_epoch        
        optimizer = self.optimizer
        network = self.network

        # Gluon-CV requires you to create and load the parameters of your model first on 
        # the CPU - so specify ctx=None - and when all that is done you move the 
        # whole model on the GPU with:
        self.net.collect_params().reset_ctx(self.ctx)

        # First create the trainer. Obs: you should reset_ctx before creating the optimizer
        self.create_optimizer()

        # speeds up the training process
        # Check: https://mxnet.apache.org/api/python/docs/tutorials/performance/backend/amp.html
        # trainer = self.trainer
        # amp.init_trainer(trainer)

        if network == 'ssd':
            self.ssd_train()
        elif network == 'yolo':
            self.yolo_train()

if __name__ == '__main__':
    try:
        neptune.init('caioviturino/IJR2020')

        PARAMS = {'model_name': 'yolo3_darknet53_coco',
                  'ctx': 'gpu',
                  'lr_decay_epoch': '30,50',
                  'lr': 0.00001, # 0.001,
                  'lr_decay': 0.1,
                  'batch_size': 8,
                  'epochs': 80,
                  'optimizer': 'adam', #https://mxnet.apache.org/versions/1.6/api/python/docs/tutorials/packages/optimizer/index.html
                  'wd': 0.0005, # 0.0005, # sgd parameter
                  'momentum': 0.9, # 0.9, # sgd parameter
                  'beta1': 0.9, # 0.9, # adam parameter
                  'beta2': 0.999, #0.999, # adam parameter
                  'epsilon': 1e-08, # 1e-08, # adam parameter
                  'validation_threshold': 0.5,
                  'nms_threshold': 0.5
                  }

        # create experiment (all parameters are optional)
        experiment = neptune.create_experiment(name=PARAMS['model_name'],
                                               params=PARAMS,
                                               tags=[PARAMS['model_name'], PARAMS['optimizer'], 'kleber'])

        train_object = training_network(model=PARAMS['model_name'], ctx=PARAMS['ctx'], \
                                            lr_decay_epoch=PARAMS['lr_decay_epoch'], lr=PARAMS['lr'], \
                                            lr_decay=PARAMS['lr_decay'], batch_size=PARAMS['batch_size'], epochs=PARAMS['epochs'], \
                                            optimizer=PARAMS['optimizer'], exp=experiment, \
                                            validation_threshold=PARAMS['validation_threshold'], nms_threshold=PARAMS['nms_threshold'], \
                                            beta1=PARAMS['beta1'], beta2=PARAMS['beta2'], epsilon=PARAMS['epsilon'], \
                                            wd=PARAMS['wd'], momentum=PARAMS['momentum'])

        train_object.get_dataset()
        # train_object.show_summary()

        # Loads the dataset according to the batch size and num_workers
        train_object.get_dataloader()

        # training
        train_object.train()
    except:
        raise
    finally:
        neptune.stop()