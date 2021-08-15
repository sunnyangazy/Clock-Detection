from gluoncv import model_zoo, data, utils
from matplotlib import pyplot as plt

"""
针对油气站场压力表目标识别
利用GluonCV预训练模型比较 SSD YOLO R-CNN的识别效果
By SlowNorth 21-8-14
"""

#下载预训练好的网络模型
# net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
net = model_zoo.get_model('ssd_512_resnet50_v1_coco', pretrained=True)
# net = model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=True)

im_fname = ('images/j.jpg')

x, img = data.transforms.presets.yolo.load_test(im_fname, short=512)
print('Shape of pre-processed image:', x.shape)
 
class_IDs, scores, bounding_boxs = net(x)
 
ax = utils.viz.plot_bbox(img, bounding_boxs[0], scores[0],
                         class_IDs[0], class_names=net.classes)
plt.show()