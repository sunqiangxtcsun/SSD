import os
import cv2
import sys
import numpy as np

#定义caffe路径
caffe_root = "/home/sunqiang/caffe-ssd"

#当前的额工作环境切换到caffe下面来，
# 因为caffe-ssd并没有添加到环境变量中需要将当前的环境切换到caffe-ssd的目录下
os.chdir(caffe_root)
#将caffe的路径添加到python环境下，pycaffe的环境添加到系统路径下，pycaffe编译出来的接口在caffe的python目录下
sys.path.insert(0,os.path.join(caffe_root, 'python'))

import caffe

#设置cafffe运行所需要的设备号
#caffe.set_device(0)
#caffe.set_mode_gpu()

#注意这里切换了工作环境，在设置路径的时候需要绝对路径，但过长的路径往往会导致找不到
#网络配置文件
model_def = "/home/sunqiang/deploy.prototxt"
#网络权值
model_weight = "/home/sunqiang/VGG_widerface_SSD_300x300_iter_200.caffemodel"

img_path = "/home/sunqiang/30_Surgeons_Surgeons_30_90.jpg"

#初始化网络，第三个参数表明当前网络是测试状态，只做前向运算
net = caffe.Net(model_def,model_weight,caffe.TEST)

image_data = caffe.io.load_image(img_path)
#输入的格式是rgb，需要的是bgr，需要在一次变换，通过caffe提供的接口完成变换
#blobs['data']是通过layer_name获取layer对应的blob。
#拿到blob后，取出这个blob中的data变量，通过Python接口调用后，它会自动被转成ndarray类型，这个类型本身就自带shape函数。
tranformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})#提取第一层的维度
#将数据channel提前
tranformer.set_transpose('data', (2,0,1))#----->1*3*h*w
#对数据的channel再进行变化,channel需要的数据是rgb，但输入进来的数据channel通道是bgr
tranformer.set_channel_swap('data',(2,1,0))


#减均值操作均值分别为104,117,128
tranformer.set_mean('data',np.array([104,117,123]))
#将数据的尺度规模定义到0-255
tranformer.set_raw_scale('data', 255)

#通过preprocess函数对数据进行转换
tranformer_image = tranformer.preprocess('data', image_data)
#对网络blobs的数据层进行reshape
net.blobs['data'].reshape(1,3,300,300)
#对数据层进行赋值
net.blobs['data'].data[...] = tranformer_image
#定义前向运算得到的最终的网络的输出结果，取出detection_out层的结果
detect_out = net.forward()['detection_out']

print detect_out
#得到的结果是一个7维的值：
#第一维度是当前图片在当前batch_size中的索引
#第二维度是label
#第三维度 置信读
#接下来的四个维度是矩形框左上角坐标和右下角坐标
det_label = detect_out[0,0,:,1]
det_conf  = detect_out[0,0,:,2]

det_xmin = detect_out[0,0,:,3]
det_ymin = detect_out[0,0,:,4]
det_xmax = detect_out[0,0,:,5]
det_ymax = detect_out[0,0,:,6]
#根据置信度对人脸框进行筛选，去掉些置信度低的人脸框
top_indices = [i for i , conf in enumerate(det_conf) if conf >=0.1]
#根据保存下的索引，取出相应的框的置信读
top_conf = det_conf[top_indices]
#同样根据相应的索引值取出相应的坐标
top_xmin = det_xmin[top_indices]
top_ymin = det_ymin[top_indices]
top_xmax = det_xmax[top_indices]
top_ymax = det_ymax[top_indices]

[height,width,_] = image_data.shape
#选择前5个框进行绘制
for i in range(min(5, top_conf.shape[0])):
    #得到的坐标是归一化后的值，在画图中要回到原始尺寸
    xmin = int(top_xmin[i] * width)
    ymin = int(top_ymin[i] * height)
    xmax = int(top_xmax[i] * width)
    ymax = int(top_ymax[i] * height)

    cv2.rectangle(image_data, (xmin,ymin),(xmax,ymax),(255,0,0),5)

cv2.imshow("face", image_data)

cv2.waitKey(0)
