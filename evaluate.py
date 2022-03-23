from model import SegmentationModel
from keras.utils import plot_model
from dataset import Dataset

model_name = 'Unet'               # Unet
backbone = 'efficientnetb4'       # efficientnetb4
dual = True
log_dir = './%s_%s_logs' % (model_name, backbone)

# Load Data
filelocation = 'E:\\Dataset\\people_segmentation'
train_data = Dataset(filelocation, set='train')
valid_data = Dataset(filelocation, set='valid')

net = SegmentationModel((None, None, 3), classes=2, model_name=model_name,
                        backbone=backbone, weights='./%s/model.h5' % log_dir)
# plot_model(net.model, '%s.png'%model_name, show_layer_names=False, show_shapes=True)

net.evaluate(train_data)
net.evaluate(valid_data)