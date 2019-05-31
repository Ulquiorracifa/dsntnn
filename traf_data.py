import os
import numpy as np
import pandas as pd
import cv2
import xml.etree.ElementTree as ET
import platform

def get_data(filepath):
    all_imgs = {}

    classes_count = {}

    class_mapping = {}
    found_bg= False

    label_path = "/home/asprohy/data/traffic/train_label_fix.csv"
    class_mapping = {
        'other': 0,
        'parkingLot': 1,
        'intersection': 2,
        'keepRight': 3,
        'leftOrRight': 4,
        'busPassage': 5,
        'leftDriving': 6,
        'slow': 7,
        'motorVehicleStraightOrRight': 8,
        'attentionToPedestrians': 9,
        'aroundTheIsland': 10,
        'straightOrRight': 11,
        'noBus': 12,
        'noMotorcycle': 13,
        'noMotorVehicle': 14,
        'noNonmotorvehicle': 15,
        'noHonking': 16,
        'interchangeStraightOrTurning': 17,
        'speedLimited40': 18,
        'speedLimited30': 19,
        'Honking': 20,
    }


    datas = pd.read_csv(label_path).values
    for c in datas[:2000]:
        filename, x1, y1, _, _, x2, y2, _, _, class_name = c

        if filename not in all_imgs:
            all_imgs[filename] = {}

            img = cv2.imread(os.path.join(filepath,filename))
            (rows, cols) = img.shape[:2]
            all_imgs[filename]['filepath'] = filename
            all_imgs[filename]['width'] = cols
            all_imgs[filename]['height'] = rows
            all_imgs[filename]['bboxes'] = []
            if np.random.randint(0, 6) > 0:
                all_imgs[filename]['imageset'] = 'trainval'
            else:
                all_imgs[filename]['imageset'] = 'test'
            #分一部分做测试

        all_imgs[filename]['bboxes'].append(
            {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

    all_data = []
    for key in all_imgs:
        all_data.append(all_imgs[key])

    # make sure the bg class is last in the list
    if found_bg:
        if class_mapping['bg'] != len(class_mapping) - 1:
            key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
            val_to_switch = class_mapping['bg']
            class_mapping['bg'] = len(class_mapping) - 1
            class_mapping[key_to_switch] = val_to_switch

    return all_data, 21, class_mapping

def get_data2(input_path):
    all_imgs = []

    classes_count = {}

    class_mapping = {}

    visualise = False

    data_paths = [os.path.join(input_path,s) for s in ['VOC2012']]#'VOC2007',

    if platform.system() =='Windows':
        label_path = 'D:\Download\\train_label_fix.csv'
    else:
        datapath = '/home/asprohy/data/traffic/train_trfc'
        label_path = "/home/asprohy/data/traffic/train_label_fix.csv"
    datas = pd.read_csv(label_path).values
    print('Parsing annotation files')

    for data_path in data_paths:

        annot_path = os.path.join(input_path, 'Annotations')
        if platform.system() == 'W':
            imgs_path = os.path.join(input_path, 'Train_fix')
        else:
            imgs_path = os.path.join(input_path, 'train_traf')
        imgsets_path_trainval = os.path.join(input_path, 'ImageSets','Main','trainval.txt')
        imgsets_path_test = os.path.join(input_path, 'ImageSets','Main','test.txt')

        print('datas.size: ', datas.shape[0])

        spiltCout =int(datas.shape[0]*0.9)

        trainval_files = datas[:spiltCout,0]
        test_files = datas[spiltCout:,0]
        # try:
        #     with open(imgsets_path_trainval) as f:
        #         for line in f:
        #             trainval_files.append(line.strip() + '.jpg')
        # except Exception as e:
        #     print(e)
        #
        # try:
        #     with open(imgsets_path_test) as f:
        #         for line in f:
        #             test_files.append(line.strip() + '.jpg')
        # except Exception as e:
        #     if data_path[-7:] == 'VOC2012':
        #         # this is expected, most pascal voc distibutions dont have the test.txt file
        #         pass
        #     else:
        #         print(e)

        annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
        idx = 0
        for annot in annots:
            try:
                idx += 1

                et = ET.parse(annot)
                element = et.getroot()

                element_objs = element.findall('object')
                element_filename = element.find('filename').text
                element_width = int(element.find('size').find('width').text)
                element_height = int(element.find('size').find('height').text)

                if len(element_objs) > 0:
                    annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
                                       'height': element_height, 'bboxes': []}

                    if element_filename in trainval_files:
                        annotation_data['imageset'] = 'trainval'
                    elif element_filename in test_files:
                        annotation_data['imageset'] = 'test'
                    else:
                        annotation_data['imageset'] = 'trainval'

                for element_obj in element_objs:
                    class_name = element_obj.find('name').text
                    if class_name not in classes_count:
                        classes_count[class_name] = 1
                    else:
                        classes_count[class_name] += 1

                    if class_name not in class_mapping:
                        class_mapping[class_name] = len(class_mapping)

                    obj_bbox = element_obj.find('bndbox')
                    x1 = int(round(float(obj_bbox.find('xmin').text)))
                    y1 = int(round(float(obj_bbox.find('ymin').text)))
                    x2 = int(round(float(obj_bbox.find('xmax').text)))
                    y2 = int(round(float(obj_bbox.find('ymax').text)))
                    difficulty = int(element_obj.find('difficult').text) == 1
                    annotation_data['bboxes'].append(
                        {'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
                all_imgs.append(annotation_data)

                if visualise:
                    img = cv2.imread(annotation_data['filepath'])
                    for bbox in annotation_data['bboxes']:
                        cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
                                                                          'x2'], bbox['y2']), (0, 0, 255))
                    cv2.imshow('img', img)
                    cv2.waitKey(0)

            except Exception as e:
                print(e)
                continue
    return all_imgs, classes_count, class_mapping



def get_value(filepath):
    all_imgs = {}
    label_path = "/home/asprohy/data/traffic/train_label_fix.csv"
    datas = pd.read_csv(filepath).values
    for c in datas:
        filename, x1, y1, _, _, x2, y2, _, _, class_name = c

        all_imgs.append([filename, int((x1+x2)/2), int((y1+y2)/2), x2-x1, y2-y1, class_name])

    return all_imgs


def get_test_data():
    filepath = "/home/asprohy/data/traffic/submit_sample_fix.csv"
    datas = pd.read_csv(filepath).values
    return datas

def get_testlocal_data(filepath):
    datas = pd.read_csv(filepath).values
    return datas
#
if __name__ == '__main__':
    label_path = "/home/asprohy/data/traffic/train_label_fix.csv"
    datas = pd.read_csv(label_path).values
    print(np.mean(datas[:,5]-datas[:,1]))
    print(np.mean(datas[:,6]-datas[:,2]))

