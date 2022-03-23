import cv2, os, glob
import numpy as np
from keras.utils import to_categorical, Sequence
import imgaug.augmenters as iaa

def resize_image(image, target_hw):
    h, w = image.shape[0], image.shape[1]
    sh, sw = target_hw[0]/h, target_hw[1]/w
    scale = min(sh, sw)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if len(image.shape)==2:
        target_image = np.zeros(target_hw)
    else:
        target_image = np.zeros(target_hw+(3,))
    target_image[:image.shape[0], :image.shape[1]] = image
    return target_image

def label2rgb(lbl, colormap, img=None, alpha=0.5):
    lbl_viz = np.array(colormap)[lbl]
    if img is not None and alpha!=1:
        lbl_viz = alpha*lbl_viz + (1-alpha) * img + alpha*img*(lbl_viz==0).astype('float')
        lbl_viz = lbl_viz.astype(np.uint8)
    return lbl_viz

class Dataset(Sequence):
    def __init__(self, root_path='./dataset', input_hw=(512, 512), batch_size=8, set='all'):
        self.root_path = root_path
        self.image_path = self.root_path + '/images/'
        self.label_path = self.root_path + '/masks/'
        self.image_hw = input_hw
        self.batch_size = batch_size
        self.classes = 2
        self.set = set
        self.label_files = os.listdir(self.label_path)
        self.image_files = [self.image_path+file.replace('png', 'jpg') for file in self.label_files]
        self.label_files = [self.label_path+file for file in self.label_files]
        # self.seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5),
        #                            iaa.Affine(scale=(0.5, 2.0), rotate=(-45, 45))])
        self.seq = iaa.Sequential([iaa.Fliplr(0.5)])

        if set != 'all':
            if set == 'train':
                self.image_files, self.label_files = self.image_files[:-1000], self.label_files[:-1000]
            else:
                self.image_files, self.label_files = self.image_files[-1000:], self.label_files[-1000:]

    def __len__(self):
        return len(self.image_files) // self.batch_size

    def __getitem__(self, idx):
        batch_images, batch_labels = [], []
        idxs = np.random.choice(len(self.image_files), self.batch_size)
        for idx in idxs:
            image_file, label_file = self.image_files[idx], self.label_files[idx]
            image = cv2.imread(image_file)
            label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
            image = resize_image(image, self.image_hw)
            label = resize_image(label, self.image_hw)
            batch_images.append(image)
            batch_labels.append(label)
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        if set=='train':
            t = np.concatenate([batch_images, batch_labels[..., None]])
            t = self.seq(images=t)
            batch_images, batch_labels = t[..., :-1], t[..., -1]
        # result_vis = label2rgb((batch_labels[0]).astype(np.int), colors, batch_images[0])
        # plt.imshow(result_vis)
        # plt.show()
        batch_labels = to_categorical(batch_labels, self.classes)
        return batch_images, batch_labels
