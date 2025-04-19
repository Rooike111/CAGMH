import json
import numpy as np
import os
import scipy.io as scio


def make_index(jsonData: dict, indexDict: dict):
    """
    use coco dict data as orignial data.
    indexDict: {jsonData's key: [index_key, index_value]}
    """
    result = []
    for name in indexDict:
        data = jsonData[name]
        middle_dict = {}
        for item in data:
            if item[indexDict[name][0]] not in middle_dict:
                middle_dict.update({item[indexDict[name][0]]: [item[indexDict[name][1]]]})
            else:
                middle_dict[item[indexDict[name][0]]].append(item[indexDict[name][1]])
        result.append(middle_dict)

    return result


def chage_categories2numpy(category_ids: dict, data: dict):
    # category_ids = np.array(category_ids)
    label = []
    for item in data.values():
        class_item = [0] * len(category_ids)
        # class_item = np.array(class_item)
        for class_id in item:
            for i in range(len(class_id)):
                class_item[category_ids.index(class_id[i])] = 1
        # print(data[item])
        label.append(np.asarray(class_item))

    return label


def check_file_exist(indexDict: dict, file_path: str):
    keys = list(indexDict.keys())
    for item in keys:
        # print(indexDict[item])
        if not os.path.exists(os.path.join(file_path, item)):
            print(item)
            indexDict.pop(item)
        indexDict[item] = os.path.join(file_path, item)
    return indexDict


jsonFile = '/home/wangg/DataSet/IAPRTC12/train.json'
with open(jsonFile, "r") as f:
    jsonData = json.load(f)

val_jsonFile = '/home/wangg/DataSet/IAPRTC12/test.json'
with open(val_jsonFile, "r") as f:
    val_jsonData = json.load(f)

label = jsonData['labels']
val_label = val_jsonData['labels']

indexDict = {"samples": ["image_name", "image_labels"]}

result = make_index(jsonData, indexDict)
val_result = make_index(val_jsonData, indexDict)

categoryDict = chage_categories2numpy(label, result[0])
categoryDict = np.array(categoryDict)
val_categoryDict = chage_categories2numpy(val_label, val_result[0])
val_categoryDict = np.array(val_categoryDict)
labels = np.concatenate((categoryDict, val_categoryDict))
labels = np.delete(labels, 1767, axis=0)

indexDict_ = check_file_exist(result[0], file_path='/home/wangg/DataSet/IAPRTC12/images')
val_indexDict_ = check_file_exist(val_result[0], file_path='/home/wangg/DataSet/IAPRTC12/images')
index_1 = []
index_2 = []
for item in indexDict_.values():
    index_1.append(item)
for item in val_indexDict_.values():
    index_2.append(item)
indexs = np.concatenate((index_1, index_2))
# print(indexs[1767])
indexs = np.delete(indexs, 1767, axis=0)

captions_path = '/home/wangg/DataSet/IAPRTC12/captions'
image_name_1 = []
for i in result[0].keys():
    image_name_1.append(i[:])
image_name_2 = []
for i in val_result[0].keys():
    image_name_2.append(i[:])
image_name = np.concatenate((image_name_1, image_name_2))

captions = []

for i in range(len(image_name)):
    caps_path = os.path.join(captions_path,"annotations_complete_eng", image_name[i][:-4] + '.eng')
    
    # data = linecache.getline(caps_path, 4)
    # data = data[data.rfind('TION>'):data.rfind('</DESC')]
    # data = data.strip('</DESCRIPTION>')
    # c_path.append(caps_path)
    if not os.path.exists(caps_path):
        print(caps_path)
    else:
        with open(caps_path, "r", encoding='ISO-8859-1') as f:
            data = f.readline()
            counts = 1
            while data:
                if counts >= 4:
                    break
                data = f.readline()
                counts += 1
            # data = data[]
            data = data.replace('<DESCRIPTION>', '')
            data = data.replace('</DESCRIPTION>', '')
            # data = data.split(';')[0]
            data = data.replace(';', '')
            data = data.replace(',', '')
            captions.append(data)

labelslist = {"category": labels}
indexslist = {"index": indexs}
captionlist = {"caption": captions}

scio.savemat('IAPR/label.mat', labelslist)
scio.savemat('IAPR/index.mat', indexslist)
scio.savemat('IAPR/caption.mat', captionlist)