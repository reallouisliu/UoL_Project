from model import SegmentationModel
from dataset import Dataset
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

# Set Parameters
input_hw = (512, 512)
dual = True
model_name = 'Unet'            # Unet
backbone = 'efficientnetb4'    # efficientnetb4
weights = './weights/%s_weights_tf_dim_ordering_tf_kernels_notop.h5' % backbone
optimizer = 'adam'             # sgd or adam
learning_rate = 1e-3
batch_size = 2
epochs = 100

# Load Data
train_data = Dataset('E:\\Dataset\\people_segmentation', input_hw=input_hw, batch_size=batch_size, set='train')
valid_data = Dataset('E:\\Dataset\zpeople_segmentation', input_hw=input_hw, batch_size=batch_size, set='valid')

net = SegmentationModel(input_hw+(3,), classes=2, model_name=model_name, backbone=backbone, weights=weights)
net.compile(optimizer, learning_rate)

# Training
log_dir = './%s_%s_logs' % (model_name, backbone)
net.model.load_weights('./%s/model.h5' % log_dir)
net.train(train_data, valid_data, epochs, log_dir=log_dir)