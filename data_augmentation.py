import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image
from argparse import ArgumentParser

import imgaug as ia
from imgaug import augmenters as iaa


ia.seed(1)

def parse_arguments():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-i', '--path_to_input_dataset', type=str, required=False, 
                        default="/home/crystal/work/dataset/data_split/",
                        help="path of dataset for augmentation ")
    args.add_argument('-o', '--path_to_output_dataset', type=str, required=False, 
                        default="/home/crystal/work/dataset/data_split_aug/",
                        help="path to dataset for saving after augmentation ")
    return parser.parse_args()

def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin,ymin,xmax,ymax])
        # print(bndboxlist)

    bndbox = root.find('object').find('bndbox')
    return bndboxlist


def change_xml_list_annotation(root, image_id, new_target, saveroot,id):

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    index = 0
    filename = xmlroot.findall('filename')#liumm 改一下xml内的图片名字，要不然xml文件名字改了，文件里面的图片名却没改
    filename[0].text = str(image_id) + "_aug_" + str(id) + '.jpg'#liumm

    size = xmlroot.findall('size')[0]  #liumm
    width = int(size.find("width").text) #liumm
    height = int(size.find("height").text) #liumm

    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]
        index = index + 1
        if new_xmin < 0 or new_ymin < 0 or new_xmax > width or new_ymax > height: #liumm
            xmlroot.remove(object)
            continue #liumm
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

    tree.write(os.path.join(saveroot, str(image_id) + "_aug_" + str(id) + '.xml'))



if __name__ == "__main__":

    args = parse_arguments()
    folder = ["train", "valid", "test"]
    imgxml = ["images", "xmls"]
    datapath = args.path_to_input_dataset
    savepath = args.path_to_output_dataset


    for f in folder:
        for ix in imgxml:
            if not os.path.exists(os.path.join(savepath, f, ix)):
                os.makedirs(os.path.join(savepath, f, ix))
        xml_path = os.path.join(datapath, f, "xmls/")
        xml_path_aug = os.path.join(savepath, f, "xmls/")
        
        AUGLOOP = 10 # 每张影像增强的数量

        boxes_img_aug_list = []
        new_bndbox = []
        new_bndbox_list = []

        # 影像增强
        seq = iaa.Sequential([
            iaa.Flipud(0.5),  # vertically flip 20% of all images
            iaa.Fliplr(0.5),  # 镜像
            iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
            iaa.GaussianBlur(sigma=(0, 3.0)), # iaa.GaussianBlur(0.5),
            iaa.Affine(
                translate_px={"x": 15, "y": 15},
                scale=(0.8, 0.95),
                rotate=(-30, 30)
            )  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
        ])

        for root, sub_folders, files in os.walk(xml_path):
            for name in files:
                bndbox = read_xml_annotation(xml_path, name)

                for epoch in range(AUGLOOP):
                    seq_det = seq.to_deterministic()  # 保持坐标和图像同步改变，而不是随机

                    # 读取图片
                    img = Image.open(os.path.join(xml_path.replace("/xmls/","/images/"), name[:-4] + '.jpg'))
                    img = np.array(img)

                    # bndbox 坐标增强
                    for i in range(len(bndbox)):
                        bbs = ia.BoundingBoxesOnImage([
                            ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                        ], shape=img.shape)

                        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                        boxes_img_aug_list.append(bbs_aug)

                        # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                        new_bndbox_list.append([int(bbs_aug.bounding_boxes[0].x1),
                                                int(bbs_aug.bounding_boxes[0].y1),
                                                int(bbs_aug.bounding_boxes[0].x2),
                                                int(bbs_aug.bounding_boxes[0].y2)])
                    # 存储变化后的图片
                    image_aug = seq_det.augment_images([img])[0]
                    path = os.path.join(xml_path_aug.replace("/xmls/","/images/"), str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                    # image_auged = bbs.draw_on_image(image_aug, thickness=0)
                    Image.fromarray(image_aug).save(path)

                    # 存储变化后的XML
                    change_xml_list_annotation(xml_path, name[:-4], new_bndbox_list,xml_path_aug,epoch)
                    #print(str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                    new_bndbox_list = []
