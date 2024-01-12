import cv2
import random
import json, os
from pycocotools.coco import COCO
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from yolox.data import COCODataset_oriented, TrainTransform_oriented, YoloBatchSampler, DataLoader, InfiniteSampler, worker_init_reset_seed, DataPrefetcher, MosaicDetection_oriented

train_json = 'D:/program/YOLOX/DOTA_devkit/example/DOTA_test.json'
train_path = 'D:/program/YOLOX/DOTA_devkit/example/images/'

images_path = 'D:/program/YOLOX/DOTA_devkit/example'
jsonname = 'DOTA_test.json'


def visualization_bbox_seg(num_image, json_path, img_path, *str):  # 需要画图的是第num副图片， 对应的json路径和图片路径

    coco = COCO(json_path)

    if len(str) == 0:
        catIds = []
    else:
        catIds = coco.getCatIds(catNms=[str[0]])  # 获取给定类别对应的id 的dict（单个内嵌字典的类别[{}]）
        catIds = coco.loadCats(catIds)[0]['id']  # 获取给定类别对应的id 的dict中的具体id

    list_imgIds = coco.getImgIds(catIds=catIds)  # 获取含有该给定类别的所有图片的id
    img = coco.loadImgs(list_imgIds[num_image - 1])[0]  # 获取满足上述要求，并给定显示第num幅image对应的dict
    image_name = img['file_name']  # 读取图像名字
    image_id = img['id']  # 读取图像id

    image = io.imread(img_path + img['file_name'])  # 读取图像
    img_annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)  # 读取这张图片的所有seg_id
    img_anns = coco.loadAnns(img_annIds)

    for i in range(len(img_annIds)):
        min_x, min_y, w, h = img_anns[i - 1]['bbox']  # 读取边框
        center_x = min_x + w / 2
        center_y = min_y + h / 2
        Plus90 = 0
        if img_anns[i - 1]['rotate'] >= 90:
            Plus90 = 1
            temp = h
            h = w
            w = temp
        center = (center_x, center_y)
        rect_min = (center, (w,h),(img_anns[i - 1]['rotate']-Plus90*90))
        bbox = cv2.boxPoints(rect_min)
        bbox = np.int0(bbox)
        image = cv2.drawContours(image, [bbox], 0, (0, 255, 255), 2)

    # plt.rcParams['figure.figsize'] = (20.0, 20.0)
    plt.imshow(image)
    # coco.showAnns(img_anns)
    plt.show()

def visualization_augment_oriented(count_num, json_file, img_path, input_size, cache: bool = False):
    my_data = COCODataset_oriented(
        data_dir=img_path,
        json_file=json_file,
        name="images",
        img_size=input_size,
        preproc=TrainTransform_oriented(
            max_labels=10,
            flip_prob=1,
            hsv_prob=1
        ),
        cache=cache
    )

    my_data = MosaicDetection_oriented(
        dataset=my_data,
        mosaic=True,
        img_size=input_size,
        preproc=TrainTransform_oriented(
            max_labels=1000,
            flip_prob=1.0,
            hsv_prob=1.0),
        degrees=10.0,
        translate=0.2,
        # mosaic_scale=(0.8, 1.8), #常规mosaic（先mosaic再仿射）这种范围比较合适
        mosaic_scale=(2, 3),       #简单mosaic（先仿射再mosaic）这种范围比较合适
        mixup_scale=(1.0 , 3.0),
        shear=10,
        enable_mixup=True,
        mosaic_prob=1.0,
        mixup_prob=1.0,
    )

    sampler = InfiniteSampler(len(my_data), seed= 0)

    batch_sampler = YoloBatchSampler(
        sampler=sampler,
        batch_size=1,
        drop_last=False,
        mosaic=not False,
    )

    dataloader_kwargs = {"num_workers": 1, "pin_memory": True}
    dataloader_kwargs["batch_sampler"] = batch_sampler

    # Make sure each process has different random seed, especially for 'fork' method.
    # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
    dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

    train_loader = DataLoader(my_data, **dataloader_kwargs)
    prefetcher = DataPrefetcher(train_loader)
    for i in range(count_num):
        inps, targets = prefetcher.next()
        my_image = np.squeeze(inps.cpu().numpy())
        my_target = np.squeeze(targets.cpu().numpy())
        my_image = my_image.transpose(1,2,0)
        image = np.asarray(my_image.astype(np.uint8), order='C')
        visualization_bbox(image, my_target)


def visualization_bbox(my_image, my_target):
    num= np.size(my_target, 0)
    for i in range(num):
        target_t = my_target[i]
        bbox = target_t[1:5]
        if bbox[0] == 0:
            continue
        center_x, center_y, w, h = bbox  # 读取边框
        rotate = target_t[5]
        Plus90 = 0
        if rotate > 90:
            Plus90 = 1
            temp = h
            h = w
            w = temp
        center = (center_x, center_y)
        rect_min = (center, (w,h),(rotate-Plus90*90))
        bbox = cv2.boxPoints(rect_min)
        bbox = bbox.astype(int)
        my_image = cv2.drawContours(my_image, [bbox], 0, (0, 255, 255), 2)
    plt.imshow(my_image)
    plt.show()

if __name__ == "__main__":
    # visualization_bbox_seg(1, train_json, train_path, 'ship')

    visualization_augment_oriented(30, jsonname, images_path, (1280, 1280))
