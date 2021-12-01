# -*- coding: utf-8 -*-
import os
from PIL import Image
import time
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
# plt.switch_backend('Agg')
from utils import label_map_util

from utils import visualization_utils as vis_util

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_arguments():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-i', '--path_to_test_images', type=str, required=False, 
                        default="/home/username/work/dataset/data_split_aug/test/images/",
                        help="path of the images folder used for test ")
    args.add_argument('-m', '--path_to_pb', type=str, required=False, 
                        default="/home/username/work/train_log/frozen_inference_graph.pb",
                        help="path of pb model ")
    args.add_argument('-l', '--path_to_labels', type=str, required=False, 
                        default="/home/username/work/train_log/label_map.pbtxt",
                        help="path of label map ")
    args.add_argument('-r', '--path_to_results', type=str, required=False, 
                        default="/home/username/work/dataset/data_split_aug/test_results/",
                        help="path of results images ")
    args.add_argument('-c', '--num_classes', type=int, required=False, 
                        default=1,
                        help="num of classes ")
    return parser.parse_args()

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def drawGT(img_path, image, test_image_name):
    import PIL.Image as Image
    import PIL.ImageDraw as ImageDraw
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw = ImageDraw.Draw(image_pil)
    #(im_width, im_height) = image.size
    import xml.etree.ElementTree as ET
    xml_path = img_path.replace("images","xmls") + test_image_name.split('.')[0] + ".xml"
    in_file = open(xml_path)  
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    num_object = 0
    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        num_object += 1
        bndbox = object.find('bndbox')  # 子节点下节点rank的值
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text
        (left, right, top, bottom) = (int(xmin), int(xmax), int(ymin), int(ymax))
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                    (left, top)],
                    width=10,
                    fill='red')
    
    np.copyto(image, np.array(image_pil))
    return num_object




if __name__ == '__main__':
    args = parse_arguments()
    img_path = args.path_totest_images

    IMAGE_SIZE = (12, 8)
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        # loading ckpt file to graph
        with tf.gfile.GFile(args.path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    # Loading label map
    label_map = label_map_util.load_labelmap(args.path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=args.num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    # Helper code
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            start = time.time()
            for test_image in os.listdir(img_path):
                image = Image.open(img_path + test_image)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

                ### liumm
                num_object = drawGT(img_path, image_np, test_image)

                ### liumm
                final_score = np.squeeze(scores)
                count = 0
                for i in range(100):
                    if scores is None or final_score[i] > 0.5:
                        count = count + 1
                #print()
                #print("the count of objects is: ", count)
                if num_object != count:
                    print(test_image, " num_object: ", num_object, "  count: ", count)
                (im_width, im_height) = image.size
                for i in range(count):
                    # print(boxes[0][i])
                    y_min = boxes[0][i][0] * im_height
                    x_min = boxes[0][i][1] * im_width
                    y_max = boxes[0][i][2] * im_height
                    x_max = boxes[0][i][3] * im_width
                    x = int((x_min + x_max) / 2)
                    y = int((y_min + y_max) / 2)
                    if category_index[classes[0][i]]['name'] == "tower":
                        print("this image has a tower!")
                        y = int((y_max - y_min) / 4 * 3 + y_min)
                    #print("object{0}: {1}".format(i, category_index[classes[0][i]]['name']),
                    #      ',Center_X:', x, ',Center_Y:', y)
                    # print(x_min,y_min,x_max,y_max)
                plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(image_np)
                picName = test_image.split('/')[-1]
                # print(picName)
                plt.savefig(args.path_to_results + picName)
                #print(test_image + ' succeed')

            end = time.time()
            seconds = end - start
            print("Time taken : {0} seconds".format(seconds))





