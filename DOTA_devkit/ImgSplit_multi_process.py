"""
-------------
This is the multi-process version
"""
import os
import codecs
import numpy as np
import math
from dota_utils import GetFileFromThisRootDir
import cv2
import shapely.geometry as shgeo
import dota_utils as util
import copy
from multiprocessing import Pool
from functools import partial
import itertools
import time

def choose_best_pointorder_fit_another(poly1, poly2):
    """
        To make the two polygons best fit with each point
    """
    x1 = poly1[0]
    y1 = poly1[1]
    x2 = poly1[2]
    y2 = poly1[3]
    x3 = poly1[4]
    y3 = poly1[5]
    x4 = poly1[6]
    y4 = poly1[7]
    combinate = [np.array([x1, y1, x2, y2, x3, y3, x4, y4]), np.array([x2, y2, x3, y3, x4, y4, x1, y1]),
                 np.array([x3, y3, x4, y4, x1, y1, x2, y2]), np.array([x4, y4, x1, y1, x2, y2, x3, y3])]
    dst_coordinate = np.array(poly2)
    distances = np.array([np.sum((coord - dst_coordinate)**2) for coord in combinate])
    sorted = distances.argsort()
    return combinate[sorted[0]]

def cal_line_length(point1, point2):
    return math.sqrt( math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def split_single_warp(name, split_base, rate, extent):
    split_base.SplitSingle(name, rate, extent)

def split_single_by_object_warp(name, split_base, rate, extent):
    split_base.SplitSingle_by_object(name, rate, extent)
class splitbase():
    def __init__(self,
                 basepath,
                 outpath,
                 code = 'utf-8',
                 gap=512,
                 subsize=1024,
                 thresh=0.7,
                 choosebestpoint=True,
                 ext = '.png',
                 padding=True,
                 num_process=8,
                 gta_min = 0
                 ):
        """
        :param basepath: base path for dota data
        :param outpath: output base path for dota data,
        the basepath and outputpath have the similar subdirectory, 'images' and 'labelTxt'
        :param code: encodeing format of txt file
        :param gap: overlap between two patches
        :param subsize: subsize of patch
        :param thresh: the thresh determine whether to keep the instance if the instance is cut down in the process of split
        :param choosebestpoint: used to choose the first point for the
        :param ext: ext for the image format
        :param padding: if to padding the images so that all the images have the same size
        """
        self.basepath = basepath
        self.outpath = outpath
        self.code = code
        self.gap = gap
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.thresh = thresh
        self.imagepath = os.path.join(self.basepath, 'images')
        self.labelpath = os.path.join(self.basepath, 'labelTxt')
        self.outimagepath = os.path.join(self.outpath, 'images_sp')
        self.outlabelpath = os.path.join(self.outpath, 'labelTxt_sp')
        self.choosebestpoint = choosebestpoint
        self.ext = ext
        self.padding = padding
        self.num_process = num_process
        self.pool = Pool(num_process)
        self.gta_min = gta_min
        print('padding:', padding)

        # pdb.set_trace()
        if not os.path.isdir(self.outpath):
            os.mkdir(self.outpath)
        if not os.path.isdir(self.outimagepath):
            # pdb.set_trace()
            os.mkdir(self.outimagepath)
        if not os.path.isdir(self.outlabelpath):
            os.mkdir(self.outlabelpath)
        # pdb.set_trace()
    ## point: (x, y), rec: (xmin, ymin, xmax, ymax)
    # def __del__(self):
    #     self.f_sub.close()
    ## grid --> (x, y) position of grids
    def polyorig2sub(self, left, up, poly):
        polyInsub = np.zeros(len(poly))
        for i in range(int(len(poly)/2)):
            polyInsub[i * 2] = int(poly[i * 2] - left)
            polyInsub[i * 2 + 1] = int(poly[i * 2 + 1] - up)
        return polyInsub

    def calchalf_iou(self, poly1, poly2):
        """
            It is not the iou on usual, the iou is the value of intersection over poly1
        """
        inter_poly = poly1.intersection(poly2)
        inter_area = inter_poly.area
        poly1_area = poly1.area
        half_iou = inter_area / poly1_area
        return inter_poly, half_iou

    def saveimagepatches(self, img, subimgname, left, up):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.subsize, self.subsize, 3))
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)

    def saveimagepatches_by_object(self, img, subimgname, left, up, right, down):
        subimg = copy.deepcopy(img[up: down, left: right])
        outdir = os.path.join(self.outimagepath, subimgname + self.ext)
        h, w, c = np.shape(subimg)
        if (self.padding):
            outimg = np.zeros((self.subsize, self.subsize, 3))
            outimg[0:h, 0:w, :] = subimg
            cv2.imwrite(outdir, outimg)
        else:
            cv2.imwrite(outdir, subimg)

    def GetPoly4FromPoly5(self, poly):
        distances = [cal_line_length((poly[i * 2], poly[i * 2 + 1] ), (poly[(i + 1) * 2], poly[(i + 1) * 2 + 1])) for i in range(int(len(poly)/2 - 1))]
        distances.append(cal_line_length((poly[0], poly[1]), (poly[8], poly[9])))
        pos = np.array(distances).argsort()[0]
        count = 0
        outpoly = []
        while count < 5:
            #print('count:', count)
            if (count == pos):
                outpoly.append((poly[count * 2] + poly[(count * 2 + 2)%10])/2)
                outpoly.append((poly[(count * 2 + 1)%10] + poly[(count * 2 + 3)%10])/2)
                count = count + 1
            elif (count == (pos + 1)%5):
                count = count + 1
                continue

            else:
                outpoly.append(poly[count * 2])
                outpoly.append(poly[count * 2 + 1])
                count = count + 1
        return outpoly

    def savepatches(self, resizeimg, objects, subimgname, left, up, right, down):
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        mask_poly = []
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])
        with codecs.open(outdir, 'w', self.code) as f_out:
            for obj in objects:
                gtpoly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                         (obj['poly'][2], obj['poly'][3]),
                                         (obj['poly'][4], obj['poly'][5]),
                                         (obj['poly'][6], obj['poly'][7])])
                if (gtpoly.area <= self.gta_min):
                    continue
                inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)

                # print('writing...')
                if (half_iou == 1):
                    polyInsub = self.polyorig2sub(left, up, obj['poly'])
                    outline = ' '.join(list(map(str, polyInsub)))
                    outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    f_out.write(outline + '\n')
                # elif (half_iou > 0):
                elif (half_iou > self.thresh):
                  ##  print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                    out_poly = list(inter_poly.exterior.coords)[0: -1]
                    if len(out_poly) < 4:
                        continue

                    out_poly2 = []
                    for i in range(len(out_poly)):
                        out_poly2.append(out_poly[i][0])
                        out_poly2.append(out_poly[i][1])

                    if (len(out_poly) == 5):
                        #print('==========================')
                        out_poly2 = self.GetPoly4FromPoly5(out_poly2)
                    elif (len(out_poly) > 5):
                        """
                            if the cut instance is a polygon with points more than 5, we do not handle it currently
                        """
                        continue
                    if (self.choosebestpoint):
                        out_poly2 = choose_best_pointorder_fit_another(out_poly2, obj['poly'])

                    polyInsub = self.polyorig2sub(left, up, out_poly2)

                    for index, item in enumerate(polyInsub):
                        if (item <= 1):
                            polyInsub[index] = 1
                        elif (item >= self.subsize):
                            polyInsub[index] = self.subsize
                    outline = ' '.join(list(map(str, polyInsub)))
                    if (half_iou > self.thresh):
                        outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    else:
                        ## if the left part is too small, label as '2'
                        outline = outline + ' ' + obj['name'] + ' ' + '2'
                    f_out.write(outline + '\n')
                #else:
                 #   mask_poly.append(inter_poly)
        self.saveimagepatches(resizeimg, subimgname, left, up)

    def savepatches_by_object(self, resizeimg, objects, subimgname, left, up, right, down):

        this_objects = copy.deepcopy(objects)
        #先把要截取的区域图片尽量映射到给定的尺寸，也就是长边映射，短边按长边比例
        xrate = self.subsize / (right - left)
        yrate = self.subsize / (down - up)
        if xrate < yrate:
            rate = xrate
        else:
            rate = yrate
        try:
            resizeimg = cv2.resize(resizeimg, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        except Exception as e:
            print(subimgname)
            print('Error:', e)
        for obj in this_objects:
            obj['poly'] = list(map(lambda x:rate*x, obj['poly']))
        left = int(left * rate)
        right = int(right * rate)
        up = int(up * rate)
        down = int(down * rate)
        if (right - left) > self.subsize:
            wy = (right - left) - self.subsize
            right -= wy
            print(subimgname)
            print('right - left')
        if (down - up) > self.subsize:
            wy = (down - up) - self.subsize
            down -= wy
            print(subimgname)
            print('down - up')
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])

        with codecs.open(outdir, 'w', self.code) as f_out:
            for obj in this_objects:
                gtpoly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                         (obj['poly'][2], obj['poly'][3]),
                                         (obj['poly'][4], obj['poly'][5]),
                                         (obj['poly'][6], obj['poly'][7])])
                if (gtpoly.area <= 64):
                    continue
                inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)
                # print('writing...')
                if (half_iou == 1):
                    polyInsub = self.polyorig2sub(left, up, obj['poly'])
                    outline = ' '.join(list(map(str, polyInsub)))
                    outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    f_out.write(outline + '\n')
                # elif (half_iou > 0):
                elif (half_iou > self.thresh):
                  ##  print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    inter_poly = shgeo.polygon.orient(inter_poly, sign=1)
                    out_poly = list(inter_poly.exterior.coords)[0: -1]
                    if len(out_poly) < 4:
                        continue
                    #----------------------------------------------------------------------
                    # 尽量选择原来的点构建截断后的ground truth
                    original_points = list(gtpoly.exterior.coords)[0: -1]
                    interPisOrign = []
                    interPisNewInside = []
                    for pt in out_poly:
                        if pt in original_points:
                            interPisOrign.append(pt)
                        else:
                            interPisNewInside.append(pt)
                    #按面积最大保留4-len(interPisOrign) 个新点
                    itpoly_sets = []
                    itpoly_areas = []
                    for v in itertools.combinations(interPisNewInside, 4-len(interPisOrign)):
                        poly_temp = copy.deepcopy(out_poly)
                        for dn in interPisNewInside:
                            if dn not in v:
                                poly_temp.remove(dn)
                        itpoly_sets.append(poly_temp)
                        itpoly_t = shgeo.Polygon(poly_temp)
                        itpoly_areas.append(itpoly_t.area)
                    out_poly = itpoly_sets[itpoly_areas.index(max(itpoly_areas))]
                    #---------------------------------------------------------------------

                    out_poly2 = []
                    for i in range(len(out_poly)):
                        out_poly2.append(out_poly[i][0])
                        out_poly2.append(out_poly[i][1])

                    if (len(out_poly) == 5):
                        #print('==========================')
                        out_poly2 = self.GetPoly4FromPoly5(out_poly2)
                    elif (len(out_poly) > 5):
                        """
                            if the cut instance is a polygon with points more than 5, we do not handle it currently
                        """
                        continue
                    if (self.choosebestpoint):
                        out_poly2 = choose_best_pointorder_fit_another(out_poly2, obj['poly'])

                    polyInsub = self.polyorig2sub(left, up, out_poly2)

                    for index, item in enumerate(polyInsub):
                        if (item <= 1):
                            polyInsub[index] = 1
                        elif (item >= self.subsize):
                            polyInsub[index] = self.subsize
                    outline = ' '.join(list(map(str, polyInsub)))
                    if (half_iou > self.thresh):
                        outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    else:
                        ## if the left part is too small, label as '2'
                        outline = outline + ' ' + obj['name'] + ' ' + '2'
                    f_out.write(outline + '\n')
                #else:
                 #   mask_poly.append(inter_poly)
        self.saveimagepatches_by_object(resizeimg, subimgname, left, up, right, down)

    def SplitSingle(self, name, rate, extent):
        """
            split a single image and ground truth
        :param name: image name
        :param rate: the resize scale for the image
        :param extent: the image format
        :return:
        """
        img = cv2.imread(os.path.join(self.imagepath, name + extent))
        if np.shape(img) == ():
            return
        fullname = os.path.join(self.labelpath, name + '.txt')
        objects = util.parse_dota_poly2(fullname)
        for obj in objects:
            obj['poly'] = list(map(lambda x:rate*x, obj['poly']))
            # obj['poly'] = list(map(lambda x: ([2 * y for y in x]), obj['poly']))
        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation = cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'
        weight = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        left, up = 0, 0
        while (left < weight):
            if (left + self.subsize >= weight):
                left = max(weight - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                right = min(left + self.subsize, weight - 1)
                down = min(up + self.subsize, height - 1)
                subimgname = outbasename + str(left) + '___' + str(up)
                # self.f_sub.write(name + ' ' + subimgname + ' ' + str(left) + ' ' + str(up) + '\n')
                self.savepatches(resizeimg, objects, subimgname, left, up, right, down)
                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= weight):
                break
            else:
                left = left + self.slide

    def SplitSingle_by_object(self, name, rate, extent):
        """
            split a single image and ground truth according to different objects
        :param name: image name
        :param rate: the resize scale for the image
        :param extent: the image format
        :return:
        """
        obj_num = 30   #选择目标的最多个数
        img = cv2.imread(os.path.join(self.imagepath, name + extent))
        width = img.shape[1]
        height = img.shape[0]
        if np.shape(img) == ():
            return
        fullname = os.path.join(self.labelpath, name + '.txt')
        objects = util.parse_dota_poly2(fullname)
        object_names = ()
        object_sep = []
        for obj in objects:
            if obj['name'] not in object_names:
                object_names = object_names + (obj['name'],)
                object_sep.append([])
            object_sep[object_names.index(obj['name'])].append(obj)

        resizeimg = img
        outbasename = name + '__' + str(rate) + '__'

        #每一种目标，都选取最多obj_num个数的在图像里的目标，并截取这些目标所在的图片区域
        for object_one in object_sep:
            xmin = []
            ymin = []
            xmax = []
            ymax = []
            point0 = []
            obj_areas_sqrt = []
            count_num= 0
            for obj_one in object_one:
                if count_num > obj_num:
                    break
                xmin_t, ymin_t, xmax_t, ymax_t = min(obj_one['poly'][0::2]), min(obj_one['poly'][1::2]), \
                                                 max(obj_one['poly'][0::2]), max(obj_one['poly'][1::2])
                if (xmax_t > width) or (ymax_t > height):
                    continue
                count_num += 1
                obj_poly = shgeo.Polygon([(obj_one['poly'][0], obj_one['poly'][1]),
                                          (obj_one['poly'][2], obj_one['poly'][3]),
                                          (obj_one['poly'][4], obj_one['poly'][5]),
                                          (obj_one['poly'][6], obj_one['poly'][7])])
                obj_areas_sqrt.append(np.sqrt(obj_poly.area))
                point0.append(obj_poly.centroid)

                xmin.append(xmin_t)
                ymin.append(ymin_t)
                xmax.append(xmax_t)
                ymax.append(ymax_t)
            #如果没有一个点存在，那就返回
            if len(point0) == 0:
                continue
            #考虑到一些目标分得太散，使得截图后太小，因此按照目标位置，选取目标比较集中的区域，丢弃分散的目标。丢弃系数是8
            del_par = 8
            while(True):
                mean_oas = np.mean(obj_areas_sqrt)
                point_dist = []
                for this_point in point0:
                    dist_t = 0
                    for that_point in point0:
                        dist_t += this_point.distance(that_point)
                    if len(point0)<2:
                        point_dist.append(0)
                    else:
                        point_dist.append(dist_t/(len(point0)-1))
                point_dist_max = max(point_dist)
                if (point_dist_max > mean_oas * del_par) and (len(point0) > 2):
                    del_index = point_dist.index(point_dist_max)
                    del point0[del_index]
                    del xmin[del_index]
                    del ymin[del_index]
                    del xmax[del_index]
                    del ymax[del_index]
                    del obj_areas_sqrt[del_index]
                else:
                    break

            left = max([0, min(xmin)-10])
            up = max([0, min(ymin)-10])
            right = min([max(xmax)+10, width-1])
            down = min([max(ymax)+10, height-1])
            #选择尽可能补全矩形
            deta_pix = np.int0(((right - left) - (down - up))/2)
            if deta_pix > 0:
                up = max([0, up - deta_pix])
                down = min([down + deta_pix, height-1])
            else:
                left = max([0, left + deta_pix])
                right = min([right - deta_pix, width-1])

            subimgname = outbasename + obj_one['name']
            self.savepatches_by_object(resizeimg, objects, subimgname, left, up, right, down)

    def splitdata(self, rate):
        """
        :param rate: resize rate before cut
        """
        imagelist = GetFileFromThisRootDir(self.imagepath)
        imagenames = [util.custombasename(x) for x in imagelist if (util.custombasename(x) != 'Thumbs')]
        if self.num_process == 1:
            for name in imagenames:
                self.SplitSingle(name, rate, self.ext)
        else:
            # worker = partial(self.SplitSingle, rate=rate, extent=self.ext)
            worker = partial(split_single_warp, split_base=self, rate=rate, extent=self.ext)
            self.pool.map(worker, imagenames)

    def splitdata_by_object(self, rate):
        """
        :param rate: resize rate before cut
        """
        imagelist = GetFileFromThisRootDir(self.imagepath)
        imagenames = [util.custombasename(x) for x in imagelist if (util.custombasename(x) != 'Thumbs')]
        if self.num_process == 1:
            for name in imagenames:
                self.SplitSingle_by_object(name, rate, self.ext)
        else:
            worker = partial(split_single_by_object_warp, split_base=self, rate=rate, extent=self.ext)
            self.pool.map(worker, imagenames)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

class split_correct_bbox(splitbase):
    def __int__(self):
        super(splitbase, self).__int__()

    def savepatches(self, resizeimg, objects, subimgname, left, up, right, down):
        outdir = os.path.join(self.outlabelpath, subimgname + '.txt')
        mask_poly = []
        imgpoly = shgeo.Polygon([(left, up), (right, up), (right, down),
                                 (left, down)])
        with codecs.open(outdir, 'w', self.code) as f_out:
            for obj in objects:
                gtpoly = shgeo.Polygon([(obj['poly'][0], obj['poly'][1]),
                                         (obj['poly'][2], obj['poly'][3]),
                                         (obj['poly'][4], obj['poly'][5]),
                                         (obj['poly'][6], obj['poly'][7])])
                if (gtpoly.area <= 0):
                    continue
                inter_poly, half_iou = self.calchalf_iou(gtpoly, imgpoly)
                #只考虑iou大于阈值的目标
                if(half_iou > self.thresh):
                    #考虑用交集的最小外接矩形，虽然有些会在边界外，但不影响对目标的判断
                    min_rect = inter_poly.minimum_rotated_rectangle
                    min_rect = shgeo.polygon.orient(min_rect, sign=1)
                    out_poly = list(min_rect.exterior.coords)[0: -1]
                    out_poly2 = []
                    for i in range(len(out_poly)):
                        out_poly2.append(out_poly[i][0])
                        out_poly2.append(out_poly[i][1])
                    if (self.choosebestpoint):
                        out_poly2 = choose_best_pointorder_fit_another(out_poly2, obj['poly'])

                    polyInsub = self.polyorig2sub(left, up, out_poly2)
                    outline = ' '.join(list(map(str, polyInsub)))
                    if (half_iou > self.thresh):
                        outline = outline + ' ' + obj['name'] + ' ' + str(obj['difficult'])
                    else:
                        ## if the left part is too small, label as '2'
                        outline = outline + ' ' + obj['name'] + ' ' + '2'
                    f_out.write(outline + '\n')
            self.saveimagepatches(resizeimg, subimgname, left, up)
if __name__ == '__main__':
    # example usage of ImgSplit
    # start = time.clock()
    # split = splitbase(r'/data/dj/dota/val',
    #                    r'/data/dj/dota/val_1024_debugmulti-process_refactor') # time cost 19s
    # # split.splitdata(1)
    # # split.splitdata(2)
    # split.splitdata(0.4)
    #
    # elapsed = (time.clock() - start)
    # print("Time used:", elapsed)

    split = splitbase(r'D:\program\YOLOX\datasets\dota',
                       r'D:\program\YOLOX\datasets\dota',
                      gap=150,
                      subsize=600,
                      num_process=8,
                      thresh=0.5,
                      gta_min=16
                      )
    # split = split_correct_bbox(r'D:\program\YOLOX\DOTA_devkit\example',
    #                    r'D:\program\YOLOX\DOTA_devkit\split_output',
    #                   gap=100,
    #                   subsize=640,
    #                   num_process=8,
    #                   thresh=0.7,
    #                   gta_min=16
    #                   )
    # split.splitdata_by_object(1)
    split.splitdata(1)

