from model import SegmentationModel
import matplotlib.pyplot as plt
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_name = 'Unet'               # Unet
backbone = 'efficientnetb4'       # efficientnetb4

log_dir = './%s_%s_logs' % (model_name, backbone)
save_dir = './dataset/%s_%s_result' % (model_name, backbone)
net = SegmentationModel((None, None, 3), classes=2, model_name=model_name,
                        backbone=backbone, weights='./%s/model.h5' % log_dir)

os.makedirs(save_dir, exist_ok=True)

# Read and Resize Image
filenames = ['/media/taoyuanshu/新加卷/dataset/person/people_segmentation/images/ballet-dance-dancer-ballerina-39572.jpg']
for filename in filenames:
    image = cv2.imread(filename)
    old_shape = (image.shape[1], image.shape[0])
    image = cv2.resize(image, (32*(image.shape[1]//32), 32*(image.shape[0]//32)))

    # Show Result
    y = net.predict(image)
    y = 255*y
    y = cv2.resize(y, old_shape, interpolation=cv2.INTER_NEAREST)
    plt.imsave('%s/%s' % (save_dir, os.path.basename(filename)), y, cmap='gray')