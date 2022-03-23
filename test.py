from model import SegmentationModel
import matplotlib.pyplot as plt
import cv2
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

model_name = 'Unet'               # Unet
backbone = 'efficientnetb4'       # efficientnetb4

log_dir = './%s_%s_logs' % (model_name, backbone)
save_dir = './dataset/%s_%s_result' % (model_name, backbone)
net = SegmentationModel((None, None, 3), classes=2, model_name=model_name,
                        backbone=backbone, weights='./%s/model.h5' % log_dir)

os.makedirs(save_dir, exist_ok=True)


# def main():
#     print("Loading the model")
#     while 1:
#         filePath = input("Image path:")
#         if filePath == "q" or filePath == 'Q':
#             print("Exit")
#             break
#         img = cv2.imread(filePath)
#         if img is None:
#             print('Error')
#         else:
#             demonstration(filePath)

def main():
    for i in range(10):
        file_path = 'C:\\Users\\19218\\Desktop\\segmentation_person\\test_data\\pictures\\'+str(i)+'.jpg'
        demonstration(file_path)


def demonstration(img_path):
    image = cv2.imread(img_path)
    old_shape = (image.shape[1], image.shape[0])
    image_resize = cv2.resize(image, (32*(image.shape[1]//32), 32*(image.shape[0]//32)))
    predict_result = net.predict(image_resize)*255
    predict_result = cv2.resize(predict_result, old_shape, interpolation=cv2.INTER_NEAREST)
    ret, mask = cv2.threshold(predict_result,0,255,cv2.THRESH_BINARY)
    result = cv2.bitwise_or(image, image, mask=mask)
    # convert OpenCV BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(18,9))
    plt.subplot(1,3,1)
    plt.title('Original')
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    plt.title('Mask')
    plt.imshow(mask)

    plt.subplot(1, 3, 3)
    plt.title('Result')
    plt.imshow(result)

    plt.show()

    # cv2.imshow('Original', image)
    # cv2.imshow('Mask', mask)
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
if __name__ == '__main__':
    main()

# # Read and Resize Image
# filenames = ['test_data/pictures/1.jpg']
# for filename in filenames:
#     image_before = cv2.imread(filename)
#     old_shape = (image_before.shape[1], image_before.shape[0])
#     image = cv2.resize(image_before, (32*(image_before.shape[1]//32), 32*(image_before.shape[0]//32)))
#     # image = cv2.resize(image_before, (1*(image_before.shape[1]//1), 1*(image_before.shape[0]//1)))
#     # image = cv2.resize(image_before,(1920,1056))
#     # Show Result
#     y = net.predict(image)
#     y = 255*y
#     y = cv2.resize(y, old_shape, interpolation=cv2.INTER_NEAREST)
#     ret, mask = cv2.threshold(y,0,255,cv2.THRESH_BINARY)
#     result = cv2.bitwise_or(image_before,image_before,mask=mask)
#     cv2.imshow('Before',image_before)
#     cv2.imshow('Mask',mask)
#     cv2.imshow('Result',result)
#     cv2.waitKey(0)
#     plt.imsave('%s/%s' % (save_dir, os.path.basename(filename)), y, cmap='gray')

