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
                 lr_decay=0.1, lr_decay_epoch='60, 80', wd=0.0005, momentum=0.9, val_interval=1, start_epoch=0,
                 epochs=2, dataset='voc', network='vgg16_atrous', save_interval=0, resume='',
                 beta1=0.9, beta2=0.999, epsilon=1e-08, validation_threshold=0.5, nms_threshold=0.5, optimizer='sgd', 
                 exp=None):
        """
        Script responsible for training the class

        Arguments:
            model (str): One of the following models [ssd_300_vgg16_atrous_voc]
            val_interval (int, default: 1): Epoch interval for validation, increase the number will reduce the
                training time if validation is slow.
            save_interval (int, default: 0): Saving parameters epoch interval, best model will always be saved.
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
        self.val_interval = val_interval
        self.start_epoch = start_epoch
        self.epochs = epochs
        self.save_interval = save_interval
        self.resume = resume
        self.validation_threshold = validation_threshold
        self.nms_threshold = nms_threshold
        self.experiment = exp
        self.beta1=beta1
        self.beta2=beta2
        self.epsilon=epsilon

        if ctx == 'cpu':
            self.ctx = [mx.cpu()]
        elif ctx == 'gpu':
            self.ctx = [mx.gpu(0)]
        else:
            raise ValueError('Invalid context.')
            
        # fix seed for mxnet, numpy and python builtin random generator.
        gutils.random.seed(233)

        self.width, self.height = dataset_commons.get_model_prop(model)
        self.model_name = model
        
        # TODO: Specify the checkpoints save path
        self.save_prefix = os.path.join(data_common['checkpoint_folder'], self.model_name)

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
        self.train_dataset = gdata.RecordFileDetection(self.train_file)
        self.val_dataset = gdata.RecordFileDetection(self.val_file)
        self.val_metric = VOC07MApMetric(iou_thresh=validation_threshold, class_names=self.net.classes)

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
        
        # use fake data to generate fixed anchors for target generation
        with mx.Context(mx.gpu(0)):
            anchors2 = anchors
        
        batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
        self.train_loader = gluon.data.DataLoader(
            train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
        
        # Val verdadeiro
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = gluon.data.DataLoader(
            val_dataset.transform(SSDDefaultValTransform(width, height)),
            batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
        
        batchify_fn_val = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
        val_loader_loss = gluon.data.DataLoader(
            val_dataset.transform(SSDCustomValTransform(width, height, anchors2)),
            batch_size, True, batchify_fn=batchify_fn_val, last_batch='rollover', num_workers=num_workers)
        
        self.val_loader = val_loader
        self.val_loader_loss = val_loader_loss

    def save_params(self, best_map, current_map, epoch, save_interval, save_param_loss, best_val_loss):
        prefix = self.save_prefix

        current_map = float(current_map)
        if current_map > best_map[0]:
            best_map[0] = current_map
            self.net.save_parameters('{:s}_best_epoch_{:04d}_map_{:.4f}.params'.format(prefix, epoch, current_map))

    def val_loss(self):
        """Training pipeline"""        
        val_data = self.val_loader_loss
        ctx = self.ctx

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

        # log the epoch info
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

        # allow the MXNet engine to perform graph optimization for best performance.
        self.net.hybridize(static_alloc=True, static_shape=True)

        # total number of correct prediction by class
        tp = [0] * len(self.classes)
        # count the number of gt by class
        gt_by_class = [0] * len(self.classes)
        # false positives by class
        fp = [0] * len(self.classes)
        # rec and prec by class
        rec_by_class = [0] * len(self.classes)
        prec_by_class = [0] * len(self.classes)

        for batch in val_data:
            batch_size = batch[0].shape[0]
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
            
            # update metric
            val_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids) #, gt_difficults)

            # Get Micro Averaging (precision and recall by each class) in each batch
            for img in range(batch_size):
                gt_ids_teste, gt_bboxes_teste = [], []
                for ids in det_ids[0][img]:
                    det_ids_number = (int(ids.asnumpy()[0]))
                    # It is required to check if the predicted class is in the image
                    # otherwise, count it as a false positive and do not include in the list
                    if det_ids_number in list(gt_ids[0][img]):
                        gt_index = list(gt_ids[0][img]).index(det_ids_number)
                        gt_ids_teste.extend(gt_ids[0][img][gt_index])
                        gt_bboxes_teste.append(gt_bboxes[0][img][gt_index])
                    else:
                        fp[det_ids_number] += 1  # Wrong classification

                xww = 1
                
                # count +1 for this class id. It will get the total number of gt by class
                # It is useful when considering unbalanced datasets
                for gt_idx in gt_ids[0][img]:
                    index = int(gt_idx.asnumpy()[0])
                    gt_by_class[index] += 1
                
                for ids in range(len(gt_bboxes_teste)):
                    det_bbox_ids = det_bboxes[0][img][ids]
                    det_bbox_ids = det_bbox_ids.asnumpy()
                    det_bbox_ids = np.expand_dims(det_bbox_ids, axis=0)
                    predict_ind = int(det_ids[0][img][ids].asnumpy()[0])
                    
                    gt_bbox_ids = gt_bboxes_teste[ids]
                    gt_bbox_ids = gt_bbox_ids.asnumpy()
                    gt_bbox_ids = np.expand_dims(gt_bbox_ids, axis=0)
                    gt_ind = int(gt_ids_teste[ids].asnumpy()[0])
                    
                    iou = bbox_iou(det_bbox_ids, gt_bbox_ids)

                    # Uncomment the following line if you want to plot the images in each inference to visually  check the tp, fp and fn 
                    # self.show_images(x, gt_bbox, det_bbox, img)
                    
                    # Check if IoU is above the threshold and the class id corresponds to the ground truth
                    if (iou > validation_threshold) and (predict_ind == gt_ind):
                        tp[gt_ind] += 1 # Correct classification
                    else:
                        fp[predict_ind] += 1  # Wrong classification
        
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
        return val_metric.get(), rec_by_class, prec_by_class

    def create_optimizer(self):
        print('aqui 1')
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

    def train(self):
        """Training pipeline"""
        train_data = self.train_loader
        val_data = self.val_loader
        eval_metric = self.val_metric
        ctx = self.ctx
        lr = self.learning_rate
        wd = self.weight_decay
        momentum = self.momentum
        lr_decay = self.lr_decay
        lr_decay_epoch = self.lr_decay_epoch
        val_interval = self.val_interval
        save_prefix = self.save_prefix
        start_epoch = self.start_epoch
        epochs = self.epochs
        save_interval = self.save_interval
        experiment = self.experiment
        optimizer = self.optimizer

        # Gluon-CV requires you to create and load the parameters of your model first on 
        # the CPU - so specify ctx=None - and when all that is done you move the 
        # whole model on the GPU with:
        self.net.collect_params().reset_ctx(self.ctx)

        # First create the trainer. Obs: you should reset_ctx before creating the optimizer
        self.create_optimizer()
        trainer = self.trainer

        # speeds up the training process
        # Check: https://mxnet.apache.org/api/python/docs/tutorials/performance/backend/amp.html
        # amp.init_trainer(trainer)

        if optimizer =='sgd':
            # lr decay policy
            lr_decay = float(lr_decay)
            lr_steps = sorted([float(ls) for ls in lr_decay_epoch.split(',') if ls.strip()])

        mbox_loss = gcv.loss.SSDMultiBoxLoss()
        ce_metric = mx.metric.Loss('CrossEntropy')
        smoothl1_metric = mx.metric.Loss('SmoothL1')

        print('Start training from [Epoch {}]'.format(start_epoch))
        
        best_map = [0]
        best_val_loss = 20
        start_train_time = time.time()

        # TODO: Speficy the summary save path
        
        # with SummaryWriter(logdir=self.path_summary) as sw:
        for epoch in range(start_epoch, epochs):
            start_epoch_time = time.time()

            if experiment:
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
            if experiment:
                experiment.log_metric('cross_entropy_training_loss', epoch, loss1)
                experiment.log_metric('smooth_l1_training_loss', epoch, loss2) 
                experiment.log_metric('train_sum_loss', epoch, loss1+loss2) 

            print('[Epoch {}] Training time (min): {:.3f}, {}={:.3f}, {}={:.3f}'.format(
                epoch, (time.time()-tic)/60, name1, loss1, name2, loss2))

            # Perform validation
            if not (epoch + 1) % val_interval:
                val_name1, val_loss1, val_name2, val_loss2 = self.val_loss()
                current_val_loss = val_loss1+val_loss2
                if experiment:
                    experiment.log_metric('cross_entropy_validation_loss', epoch, val_loss1)
                    experiment.log_metric('smooth_l1_validation_loss', epoch, val_loss2) 
                    experiment.log_metric('validation_sum_loss', epoch, current_val_loss) 

                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    save_param_loss = 1
                else:
                    save_param_loss = 0

                (map_name, mean_ap), rec_by_class, prec_by_class = self.validate()
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])

                for i, class_name in enumerate(self.classes):
                    if experiment:
                        experiment.log_metric('rec_by_class_val_' + class_name, epoch, rec_by_class[i])
                        experiment.log_metric('prec_by_class_val_' + class_name, epoch, prec_by_class[i])

                for k, v in zip(map_name, mean_ap):
                    if experiment:
                        experiment.log_metric('map_' + k, epoch, v)

                print('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                current_map = float(mean_ap[-1])
            else:
                current_map = 0.

            # self.save_params(best_map, current_map, epoch, save_interval, save_param_loss, best_val_loss)            

        # Displays the total time of the training
        end_train_time = time.time()
        print('Train time {:.3f}'.format(end_train_time - start_train_time))

if __name__ == '__main__':
    # neptune.init('caioviturino/IJR2020')
    
    PARAMS = {'model_name': 'ssd_300_vgg16_atrous_coco',
              'ctx': 'gpu',
              'lr_decay_epoch': '30,50',
              'lr': 0.00014348044333442934, # 0.001,
              'lr_decay': 0.1,
              'batch_size': 32,
              'epochs': 60,
              'save_interval': 100,
              'optimizer': 'adam', #https://mxnet.apache.org/versions/1.6/api/python/docs/tutorials/packages/optimizer/index.html
              'wd': 0.0009247124194404376, # 0.0005, # sgd parameter
              'momentum': 0.7561884086974885, # 0.9, # sgd parameter
              'beta1': 0.9072439322788038, # 0.9, # adam parameter
              'beta2': 0.9958530706682409, #0.999, # adam parameter
              'epsilon': 1.316405901710664e-08, # 1e-08, # adam parameter
              'validation_threshold': 0.5,
              'nms_threshold': 0.5
              }

    # create experiment (all parameters are optional)
    # experiment = neptune.create_experiment(name=PARAMS['model_name'],
                                        #    params=PARAMS,
                                        #    tags=[PARAMS['model_name'], PARAMS['optimizer']]) exp=experiment,

    train_object = training_network(model=PARAMS['model_name'], ctx=PARAMS['ctx'], \
                                    lr_decay_epoch=PARAMS['lr_decay_epoch'], lr=PARAMS['lr'], \
                                    lr_decay=PARAMS['lr_decay'], batch_size=PARAMS['batch_size'], epochs=PARAMS['epochs'], \
                                    save_interval=PARAMS['save_interval'], optimizer=PARAMS['optimizer'],  \
                                    validation_threshold=PARAMS['validation_threshold'], nms_threshold=PARAMS['nms_threshold'], \
                                    beta1=PARAMS['beta1'], beta2=PARAMS['beta2'], epsilon=PARAMS['epsilon'], \
                                    wd=PARAMS['wd'], momentum=PARAMS['momentum'])
    
    train_object.get_dataset()
    # train_object.show_summary()

    # Loads the dataset according to the batch size and num_workers
    train_object.get_dataloader()

    # try:
        # training
    train_object.train()
    # except:
        # print('Stopped training')
    # finally:
        # neptune.stop()