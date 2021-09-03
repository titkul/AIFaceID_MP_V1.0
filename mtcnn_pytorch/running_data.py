from src import detect_faces, show_bboxes

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2 
import imageio
import os

path_model = "models"
image_size = 112
input_datadir = "./raw"
output_dir  = '../facebank'

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_dataset(paths, has_class_directories=True):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            image_paths = get_image_paths(facedir)
            dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

bounding_boxes_filename = os.path.join(path_model, 'bounding_boxes.txt')
listpat = os.listdir(input_datadir)   
dataset = get_dataset(input_datadir)

with open(bounding_boxes_filename, "+a") as text_file:
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            output_filename = os.path.join(output_class_dir, filename + '.png')
            print("Image: %s" % image_path)
            if not os.path.exists(output_filename):
                try:
                    img = imageio.imread(image_path)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))
                        continue
                    if img.ndim == 2:
                        img = to_rgb(img)
                        print('to_rgb data dimension: ', img.ndim)
                    img = img[:, :, 0:3]
                    imgs = Image.fromarray(img)
                    bounding_boxes, landmarks = detect_faces(imgs, min_face_size=40.0)

                    nrof_faces = bounding_boxes.shape[0]
                    print('No of Detected Face: %d' % nrof_faces)
                    if nrof_faces > 0:

                        det = bounding_boxes[:, 0:4]
                        img_size = np.asarray(img.shape)[0:2]

                        bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                        img_center = img_size / 2
                        offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                                (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                        offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                        index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                        det = det[index, :]
                        det = np.squeeze(det)
                        bb_temp = np.zeros(4, dtype=np.int32)

                        bb_temp[0] = det[0]
                        bb_temp[1] = det[1]
                        bb_temp[2] = det[2]
                        bb_temp[3] = det[3]

                        if bb_temp[0] < 0:
                            bb_temp[0] = 0
                        if bb_temp[1] < 0:
                            bb_temp[1] = 0
                        if bb_temp[2] < 0:
                            bb_temp[2] = 0
                        if bb_temp[3] < 0:
                            bb_temp[3] = 0

                        #np.array(Image.fromarray(cropped[i]).resize((image_size, image_size)))
                        cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                        scaled_temp =np.array(Image.fromarray(cropped_temp).resize((image_size, image_size)))
                        nrof_successfully_aligned += 1
                        imageio.imwrite(output_filename, scaled_temp)
                        text_file.write('%s %d %d %d %d\n' % (output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))     
                    else:
                        print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))
