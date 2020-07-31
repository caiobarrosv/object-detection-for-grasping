"""Train SSD"""
# import argparse
import os
import logging
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
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.metrics.accuracy import Accuracy
from mxboard import SummaryWriter
import cv2
from gluoncv.utils.bbox import bbox_iou 
from mxnet.contrib import amp

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import utils.dataset_commons as dataset_commons
# from utils.voc_detection_custom import VOC07MApMetric

'''
The models used in this project:
    PASCAL VOC:
        1) ssd_300_vgg16_atrous_voc 
        2) ssd_512_vgg16_atrous_voc 
        3) ssd_512_resnet50_v1_voc
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
                 lr_decay=0.1, lr_decay_epoch='60, 80', wd=0.0005, momentum=0.9, val_interval=1, start_epoch=0,
                 epochs=2, dataset='voc', network='vgg16_atrous', save_interval=0, log_interval=20, resume='',
                 validation_threshold=0.5, nms_threshold=0.5):
        """
        Script responsible for training the class

        Arguments:
            model (str): One of the following models [ssd_300_vgg16_atrous_voc]
            val_interval (int, default: 1): Epoch interval for validation, increase the number will reduce the
                training time if validation is slow.
            save_interval (int, default: 0): Saving parameters epoch interval, best model will always be saved.
            log_interval (int, default: 20): Logging mini-batch interval. Default is 100.
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
            num_worker (int, default: 2): number to accelerate data loading, if you CPU and GPUs are powerful.
            dataset (str, default:'voc'): Training dataset. Now support voc.
            batch_size (int, default: 4): Training mini-batch size
            data_shape (int, default: 300): Input data shape, use 300, 512.
            network (str, default:'vgg16_atrous'): Base network name which serves as feature extraction base.
        """

        amp.init()

        # TRAINING PARAMETERS
        self.resume_training = resume_training
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.learning_rate = lr
        self.weight_decay = wd
        self.momentum = momentum
        self.optimizer = 'sgd'
        self.lr_decay = lr_decay
        self.lr_decay_epoch = lr_decay_epoch
        self.val_interval = val_interval
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume = resume
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

        if model.lower() == 'ssd300_vgg16_voc':
            self.model_name = 'ssd_300_vgg16_atrous_voc' #'ssd_300_vgg16_atrous_coco'
            self.dataset= 'voc'
            self.width, self.height = 300, 300
            self.network = 'vgg16_atrous'
        elif model.lower() == 'ssd300_vgg16_coco':
            self.model_name = 'ssd_300_vgg16_atrous_coco'
            self.dataset= 'coco'
            self.width, self.height = 300, 300
            self.network = 'vgg16_atrous'
        
        # TODO: Specify the checkpoints save path
        self.save_prefix = os.path.join(data_common['checkpoint_folder'], self.model_name)
        
        # TODO: load the train and val rec file
        self.train_file = data_common['record_train_path']
        self.val_file = data_common['record_val_path']

        classes_keys = [key for key in data_common['classes']]
        self.classes = classes_keys
        print('Classes: ', self.classes)

        # pretrained or pretrained_base?
        # pretrained (bool or str) – Boolean value controls whether to load the default 
        # pretrained weights for model. String value represents the hashtag for a certain 
        # version of pretrained weights.
        # pretrained_base (bool or str, optional, default is True) – Load pretrained base 
        # network, the extra layers are randomized. Note that if pretrained is True, this
        # has no effect.
        self.net = get_model(self.model_name, pretrained=True, norm_layer=gluon.nn.BatchNorm)
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

    def get_dataset(self):
        validation_threshold = self.validation_threshold
        nms_threshold = self.nms_threshold

        if self.dataset == 'voc':
            self.train_dataset = gdata.RecordFileDetection(self.train_file)
            self.val_dataset = gdata.RecordFileDetection(self.val_file)
            self.val_metric = VOC07MApMetric(iou_thresh=validation_threshold, class_names=self.net.classes)
        # elif dataset.lower() == 'coco':
        #     self.train_dataset = gdata.COCODetection(splits='instances_train2017')
        #     self.val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        #     self.val_metric = COCODetectionMetric(
        #         val_dataset, save_prefix + '_eval', cleanup=True,
        #         data_shape=(args.data_shape, args.data_shape))
        #     # coco validation is slow, consider increase the validation interval
        #     if args.val_interval == 1:
        #         args.val_interval = 10
        else:
            raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))

    def show_summary(self):
        self.net.summary(mx.nd.ones((1, 3, self.height, self.width)))

    def get_dataloader(self):
        batch_size, num_workers = self.batch_size, self.num_workers
        width, height = self.width, self.height
        train_dataset = self.train_dataset
        val_dataset = self.val_dataset
        batch_size = self.batch_size
        num_workers = self.num_workers

        # use fake data to generate fixed anchors for target generation
        with autograd.train_mode():
            _, _, anchors = self.net(mx.nd.zeros((1, 3, height, width)))
        
        batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
        self.train_loader = gluon.data.DataLoader(
            train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
        
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        self.val_loader = gluon.data.DataLoader(
            val_dataset.transform(SSDDefaultValTransform(width, height)),
            batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)

    def save_params(self, best_map, current_map, epoch, save_interval):
        prefix = self.save_prefix

        current_map = float(current_map)
        if current_map > best_map[0]:
            best_map[0] = current_map
            self.net.save_parameters('{:s}_best_epoch_{:04d}_map_{:.4f}.params'.format(prefix, epoch, current_map))
            with open(prefix+'_best_map.log', 'a') as f:
                f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_map))
        # if save_interval epoch % save_interval == 0:
            # self.net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))

    def show_images(self, i, data, gt_bboxes, det_bboxes, current_gt_class_id, current_pred_class_id):
        # In case you want to show the images in the validation dataset
        #     uncomment the following lines
        # for img in range(0, 4):
        gt_bbox = gt_bboxes[0][i][0].asnumpy()
        xmin_gt, ymin_gt, xmax_gt, ymax_gt = [int(x) for x in gt_bbox]
        pred_bbox = det_bboxes[0][i][0].asnumpy()
        xmin_pred, ymin_pred, xmax_pred, ymax_pred = [int(x) for x in pred_bbox]
        img = data[0][i]
        img = img.transpose((1, 2, 0))  # Move channel to the last dimension
        # img = img.asnumpy().astype('uint8') # convert to numpy array
        # img = img.astype(np.uint8)  # use uint8 (0-255)
        img = img.asnumpy()
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # OpenCV uses BGR orde
        if current_gt_class_id == 0:
            cv2.rectangle(img, (xmin_gt, ymin_gt), (xmax_gt, ymax_gt), (255, 0, 0), 1)
        else:
            cv2.rectangle(img, (xmin_gt, ymin_gt), (xmax_gt, ymax_gt), (0, 255, 0), 1)
        
        if current_pred_class_id == 0:
            cv2.rectangle(img, (xmin_pred, ymin_pred), (xmax_pred, ymax_pred), (255, 0, 0), 1)
        else:
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
        self.net.set_nms(nms_thresh=nms_threshold, nms_topk=200, post_nms=1) # default: iou=0.45 e topk=400

        # allow the MXNet engine to perform graph optimization for best performance.
        self.net.hybridize(static_alloc=True, static_shape=True)

        # total number of correct prediction by class
        tp = [0] * len(self.classes)
        # count the number of gt by class
        gt_by_class = [0] * len(self.classes)
        # false positives by class
        fp = [0] * len(self.classes)
        # false negatives by class
        fn = [0] * len(self.classes)
        # rec and prec by class
        rec_by_class = [0] * len(self.classes)
        prec_by_class = [0] * len(self.classes)

        for batch in val_data:
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []

            for x, y in zip(data, label):
                # get prediction results
                ids, scores, bboxes = self.net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                # gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            # Get Micro Averaging (precision and recall by each class)
            for i in range(0, len(gt_bboxes[0])):
                # retorna o IoU para cada um dos 4 bounding boxes (já que o batch é 4)
                iou = bbox_iou(det_bboxes[0][i].asnumpy(), gt_bboxes[0][i].asnumpy())
                # id of each one of the gt_ids
                current_gt_class_id = int(gt_ids[0][i][0].asnumpy()[0])
                current_pred_class_id = int(det_ids[0][i][0].asnumpy()[0])

                # If you want to plot the images in each inference to check the tp, fp and fn, uncomment the following line
                # self.show_images(i, data, gt_bboxes, det_bboxes, current_gt_class_id, current_pred_class_id)
                
                # count +1 for this class id. It will get the total number of gt by class
                # It is useful when considering unbalanced datasets
                gt_by_class[current_gt_class_id] += 1

                # Check if IoU is above the threshold and the class id corresponds to the ground truth
                if (iou > validation_threshold) and (current_gt_class_id == current_pred_class_id):
                    tp[current_gt_class_id] += 1
                else:
                    fp[current_pred_class_id] += 1
                    fn[current_gt_class_id] += 1

            # update metric
            val_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids) #, gt_difficults)

        tp = np.array(tp)
        fp = np.array(fp)
        # rec and prec according to the micro averaging
        for i, gt in enumerate(gt_by_class):
            rec_by_class[i] += tp[i]/gt

            # If an element of fp + tp is 0,
            # the corresponding element of prec[l] is nan.
            with np.errstate(divide='ignore', invalid='ignore'):
                prec_by_class[i] += tp[i]/(tp[i]+fp[i])

        # rec, prec = val_metric._recall_prec()
        return val_metric.get(), rec_by_class, prec_by_class

    def train(self):
        """Training pipeline"""
        train_data = self.train_loader
        val_data = self.val_loader
        eval_metric = self.val_metric
        ctx = self.ctx
        lr = self.learning_rate
        wd = self.weight_decay
        momentum = self.momentum
        optimizer = self.optimizer
        lr_decay = self.lr_decay
        lr_decay_epoch = self.lr_decay_epoch
        val_interval = self.val_interval
        save_prefix = self.save_prefix
        start_epoch = self.start_epoch
        epochs = self.epochs
        save_interval = self.save_interval
        log_interval = self.log_interval

        # Gluon-CV requires you to create and load the parameters of your model first on 
        # the CPU - so specify ctx=None - and when all that is done you move the 
        # whole model on the GPU with:
        self.net.collect_params().reset_ctx(self.ctx)

        # wd: The weight decay (or L2 regularization) coefficient.
        trainer = gluon.Trainer(self.net.collect_params(), optimizer,
                                {'learning_rate': lr, 'wd': wd, 'momentum': momentum})
        
        # speeds up the training process
        # Check: https://mxnet.apache.org/api/python/docs/tutorials/performance/backend/amp.html
        amp.init_trainer(trainer)

        # lr decay policy
        lr_decay = float(lr_decay)
        lr_steps = sorted([float(ls) for ls in lr_decay_epoch.split(',') if ls.strip()])

        mbox_loss = gcv.loss.SSDMultiBoxLoss()
        ce_metric = mx.metric.Loss('CrossEntropy')
        smoothl1_metric = mx.metric.Loss('SmoothL1')

        # set up logger
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        log_file_path = save_prefix + '_train.log'
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fh = logging.FileHandler(log_file_path)
        logger.addHandler(fh)
        # logger.info(args)
        logger.info('Start training from [Epoch {}]'.format(start_epoch))
        
        best_map = [0]
        start_train_time = time.time()

        # TODO: Speficy the summary save path
        # path_summary = './logs/teste6_rec_prec'
        path_summary = data_common['logs_folder']
        with SummaryWriter(logdir=path_summary) as sw:
            for epoch in range(start_epoch, epochs):
                start_epoch_time = time.time()

                sw.add_scalar('learning_rate', trainer.learning_rate, epoch)
                while lr_steps and epoch >= lr_steps[0]:
                    new_lr = trainer.learning_rate * lr_decay
                    lr_steps.pop(0) # removes the first element in the list
                    trainer.set_learning_rate(new_lr) # Set a new learning rate
                    logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
                
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
                        
                        with amp.scale_loss(sum_loss, trainer) as scaled_loss:
                            autograd.backward(scaled_loss)
                        # autograd.backward(sum_loss)
                    
                    # since we have already normalized the loss, we don't want to normalize
                    # by batch-size anymore
                    trainer.step(1)
                    ce_metric.update(0, [l * batch_size for l in cls_loss])
                    smoothl1_metric.update(0, [l * batch_size for l in box_loss])

                    # Log in mini-batch interval
                    if not (i + 1) % log_interval:
                        name1, loss1 = ce_metric.get()
                        name2, loss2 = smoothl1_metric.get()
                        
                        # The samples/sec should be calculated according to the batch size
                        # therefore, log_interval should be equal to the batch size
                        logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                            epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
                    btic = time.time()

                # log the epoch info
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                # MXBoard
                sw.add_scalar('epoch_loss', (name1, loss1), epoch)
                sw.add_scalar('epoch_loss', (name2, loss2), epoch)
                sw.add_scalar('mean_map_and_sum_loss', ('sum_loss', loss1+loss2), epoch)

                logger.info('[Epoch {}] Training time (min): {:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, (time.time()-tic)/60, name1, loss1, name2, loss2))

                # Perform validation
                if not (epoch + 1) % val_interval:
                    # consider reduce the frequency of validation to save time
                    (map_name, mean_ap), rec_by_class, prec_by_class = self.validate()
                    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])

                    for i, class_name in enumerate(self.classes):
                        sw.add_scalar('rec_by_class', (class_name+'_rec', rec_by_class[i]), epoch)
                        sw.add_scalar('prec_by_class', (class_name+'_prec', prec_by_class[i]), epoch)

                    sw.add_scalar('mean_map_and_sum_loss', ('mean_map', mean_ap[-1]), epoch)
                    for k, v in zip(map_name, mean_ap):
                        sw.add_scalar('map', (k, v), epoch)
                    
                    logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                    current_map = float(mean_ap[-1])
                else:
                    current_map = 0.

                self.save_params(best_map, current_map, epoch, save_interval)            

            # Displays the total time of the training
            end_train_time = time.time()
            logger.info('Train time {:.3f}'.format(end_train_time - start_train_time))

            scalars = os.path.join(data_common['logs_folder'], 'scalars.json')
            sw.export_scalars(scalars)

if __name__ == '__main__':
    train_object = training_network(model='ssd300_vgg16_voc', ctx='gpu', \
                                    lr_decay_epoch='30, 50', lr=0.001, \
                                    lr_decay=0.1, batch_size=4, epochs=60, \
                                    save_interval = 0, log_interval=1)
    
    # we have 320 images. If batch_size = 32, then, there are 10 batches
    # log_interval is related to these 10 batches
    # if log_interval=1, it will show each one of the 10 batches

    train_object.get_dataset()
    # train_object.show_summary()

    # Loads the dataset according to the batch size and num_workers
    train_object.get_dataloader()

    # training
    train_object.train()