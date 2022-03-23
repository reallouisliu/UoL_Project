from keras import optimizers, callbacks, metrics, backend as K
from dataset import label2rgb, resize_image
import segmentation_models as sm
from tqdm import tqdm
import numpy as np
import time
import cv2
import os

class SegmentationModel():
    def __init__(self, input_shape=(256, 256, 3), classes=4, model_name='Unet', backbone='vgg16', weights=None):
        self.input_shape = input_shape
        self.classes = classes
        self.model_name = model_name
        if model_name=='Unet':
            model = sm.Unet
        elif model_name=='PSPNet':
            model = sm.PSPNet
        elif model_name=='FPN':
            model = sm.FPN
        elif model_name=='Linknet':
            model = sm.Linknet
        self.model = model(input_shape=input_shape, backbone_name=backbone, classes=classes, activation='softmax',
                        encoder_weights=None)


        if os.path.exists(weights):
            print('Loading weights!')
            self.model.load_weights(weights, by_name=True, skip_mismatch=True)

    def cce_jaccard_loss(self, y_true, y_pred):
        cce = K.categorical_crossentropy(y_true, y_pred)
        cce = K.mean(cce)
        dice = self.dice_score(y_true, y_pred)
        jaccard_loss = 1-dice
        return jaccard_loss + 0.2*cce

    def dice_score(self, y_true, y_pred):
        axes = [0, 1, 2]
        intersection = K.sum(y_true * y_pred, axis=axes)
        union = K.sum(y_true + y_pred, axis=axes)
        iou = 2*intersection / (union + 1e-8)
        dice_score = K.sum(iou) / K.sum(K.cast(K.sum(y_true, axis=axes) > 0, 'float32'))
        return dice_score

    def acc(self, y_true, y_pred):
        return metrics.categorical_accuracy(y_true, y_pred)

    def compile(self, optimizer='sgd', leaning_rate=1e-3):
        if optimizer == 'sgd':
            optimizer = optimizers.SGD(leaning_rate, momentum=0.9)
        elif optimizer == 'adam':
            optimizer = optimizers.Adam(leaning_rate)
        self.model.compile(optimizer=optimizer, loss=self.cce_jaccard_loss, metrics=[self.dice_score, self.acc])

    def train(self, train_data, valid_data=None, epochs=100, log_dir='./logs'):
        log_path = '%s/%s_log.csv' % (log_dir, self.model_name)
        model_path = '%s/model.h5' % log_dir
        os.makedirs(log_dir, exist_ok=True)
        csvlogger = callbacks.CSVLogger(log_path)
        lr_recuder = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1)
        checkpoint = callbacks.ModelCheckpoint(filepath=model_path, save_best_only=True, save_weights_only=True)
        self.model.fit_generator(train_data, validation_data=valid_data, use_multiprocessing=False,
                                epochs=epochs, workers=4, verbose=1, callbacks=[csvlogger, lr_recuder, checkpoint])

    def predict(self, image):
        y = self.model.predict(image[None])
        y = np.argmax(y[0], axis=-1)
        return y.astype('uint8')

    def evaluate(self, dataset):

        if dataset.set == 'train':
            from tensorflow.python.framework import ops
            import tensorflow as tf
            g = ops.get_default_graph()
            flops = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.float_operation())  # 计算浮点运算次数
            params = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())  # 计算参数量
            print('original flops:', flops.total_float_ops)
            print('original params:', params.total_parameters)

        start = time.time()
        y_true, y_pred = [], []
        for image_file, label_file in tqdm(zip(dataset.image_files, dataset.label_files), total=len(dataset.image_files)):
            image = cv2.imread(image_file)
            label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (32 * (image.shape[1] // 32), 32 * (image.shape[0] // 32)))
            label = cv2.resize(label, (32 * (label.shape[1] // 32), 32 * (label.shape[0] // 32)))
            y = self.predict(image)
            y_true.append(label[::10, ::10].flatten().astype('uint8'))
            y_pred.append(y[::10, ::10].flatten().astype('uint8'))

        fps = len(y_true)/(time.time()-start)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        dice_coef, iou, accuracy = [], [], []
        for i in range(self.classes):
            dsc = 2*np.sum((y_pred==i)*(y_true==i))/(np.sum(y_pred==i)+np.sum(y_true==i))
            dice_coef.append(dsc)

            intersection = np.sum((y_pred==i)*(y_true==i))
            union = np.sum(y_pred==i)+np.sum(y_true==i)-intersection
            iou.append(intersection/union)

            accuracy.append(np.mean(y_pred[y_true==i]==i))
            # print('class %i --- dice_coef: %.4f, iou: %.4f, accuracy: %.2f' %
            #       (i, dice_coef[-1], iou[-1], 100*accuracy[-1]))

        mean_dice_coef = np.mean(np.array(dice_coef))
        mean_iou = np.mean(np.array(iou))
        accuracy = np.mean(y_true==y_pred)
        specificity = np.mean(y_pred[y_true==0]==0)
        sensitivity = np.mean(y_pred[y_true==1]==1)
        print('mean_dice_coef: %.4f, mean_iou: %.4f, accuracy: %.2f%%, specificity: %.2ff%%, sensitivity: %.2ff%%, fps: %.2f' %
            (mean_dice_coef, mean_iou, 100*accuracy, 100*specificity, 100*sensitivity, fps))