#按照标注目标的数目，分开不同的图片
import cv2
import os
import numpy as np
import shutil
import dota_utils as util
from dota_utils import GetFileFromThisRootDir
import matplotlib.pyplot as plt


def Seperate_imgs(path, imagefolder, labelfolder, num):
    # num 是 图片标注框的数目，大于这个数据，该图片要被重新分割，因此需要分开两种图片
    imagepath = os.path.join(path, imagefolder)
    labelpath = os.path.join(path, labelfolder)
    imagelist = GetFileFromThisRootDir(imagepath)
    imagenames = [util.custombasename(x) for x in imagelist if (util.custombasename(x) != 'Thumbs')]
    for name in imagenames:
        label_fullname = os.path.join(labelpath, name + '.txt')
        objects = util.parse_dota_poly2(label_fullname)
        if len(objects) == 0:
            continue
        img_fullname = os.path.join(imagepath, name + '.png')
        img = cv2.imread(img_fullname)
        if np.shape(img) == ():
            return
        if len(objects) > num:
            # 数据多了，把图片和label分配到同一个路径带M的文件夹下
            imagepath_new = os.path.join(path, imagefolder + 'M')
            labelpath_new = os.path.join(path, labelfolder + 'M')
        else:
            # 数据合适，把图片和label分配到同一个路径带C的文件夹下
            imagepath_new = os.path.join(path, imagefolder + 'C')
            labelpath_new = os.path.join(path, labelfolder + 'C')
        img_fullname_new = os.path.join(imagepath_new, name + '.png')
        label_fullname_new = os.path.join(labelpath_new, name + '.txt')
        if not os.path.exists(imagepath_new):
            os.mkdir(imagepath_new)
            os.mkdir(labelpath_new)
        shutil.copyfile(img_fullname, img_fullname_new)
        shutil.copyfile(label_fullname, label_fullname_new)

    return objects

def Count_objects(path, labelfolder):
    OBJECT_CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                    'basketball-court', 'storage-tank', 'soccer-ball-field',
                    'roundabout', 'harbor', 'swimming-pool', 'helicopter')
    all_objects = []
    labelpath = os.path.join(path, labelfolder)
    labellist = GetFileFromThisRootDir(labelpath)
    labelnames = [util.custombasename(x) for x in labellist if (util.custombasename(x) != 'Thumbs')]
    for name in labelnames:
        objects_in_one_img = np.zeros(15)
        label_fullname = os.path.join(labelpath, name + '.txt')
        objects = util.parse_dota_poly2(label_fullname)
        for obj in objects:
            for num, object_name in enumerate(OBJECT_CLASSES):
                if obj['name'] == object_name:
                    objects_in_one_img[num] += 1
                    break
        all_objects.append(objects_in_one_img)
    all_objects = np.asarray(all_objects)
    for i in range(15):
        plt.subplot(15, 1, i+1)
        plt.hist(all_objects[:, i], bins=300)
    plt.show()

    return all_objects

if __name__ == '__main__':
    # path = r'D:\program\原始数据集\DOTA遥感数据集\train\labelTxt-v1.0'
    path = r'D:\program\YOLOX\datasets\dota'
    # # 图片分类：按照标注的数目
    a = Seperate_imgs(path, 'images_sp', 'labelTxt_sp', 0)
    # 统计标注分析
    # a = Count_objects(path, 'labelTxt')
