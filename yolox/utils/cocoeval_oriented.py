from pycocotools.cocoeval import COCOeval
import numpy as np
import time
import copy
import cv2
import shapely.geometry as shgeo

class COCOeval_Oriented(COCOeval):
    def __int__(self):
        super(COCOeval, self).__int__()

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        self.ious = {(imgId, catId): self.computeIoU_oriented(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU_oriented(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        g = [self.bboxr2poly(g['bbox'], g['rotate']) for g in gt]
        d = [self.bboxr2poly(d['bbox'], d['rotate']) for d in dt]


        if g == [] or d == []:
            ious = []
        else:
            ious = []
            for d_ in d:
                ious_ = []
                for g_ in g:
                    ious_.append(self.poly_iou(g_, d_))
                ious.append(ious_)
            ious = np.array(ious)

        # compute iou between each dt and gt region
        # iscrowd = [int(o['iscrowd']) for o in gt]

        # ious = maskUtils.iou(d,g,iscrowd)
        return ious

    def bboxr2poly(self, bbox, rotate):
        min_x, min_y, w, h = bbox  # 读取边框
        center_x = min_x + w / 2
        center_y = min_y + h / 2
        Plus90 = 0
        if rotate >= 90:
            Plus90 = 1
            temp = h
            h = w
            w = temp
        center = (center_x, center_y)
        rect_min = (center, (w,h), (rotate-Plus90*90))
        poly = cv2.boxPoints(rect_min)
        poly_out = []
        for p in poly:
            poly_out.append((p[0],p[1]))
        poly_out = shgeo.Polygon(poly_out)
        return poly_out

    def poly_iou(self, p1, p2):
        if not p1.intersects(p2):
            iou = 0
        else:
            inter_area = p1.intersection(p2).area
            union_area = p1.area + p2.area - inter_area
            if union_area == 0:
                iou = 0
            else:
                iou = inter_area / union_area
        return iou
