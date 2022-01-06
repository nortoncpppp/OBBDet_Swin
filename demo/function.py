import re
import time
import argparse
import os
import cv2
import ogr
import osr
from osgeo import gdal
import numpy as np
import math
from demo.test_gdal import imagexy2geo,geo2imagexy,geo2lonlat
from shapely.geometry import Point,Polygon  # 多边形
from tqdm import tqdm

# import DOTA_devkit.polyiou as polyiou

def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName+"文件无法打开")
        return
    im_width = dataset.RasterXSize #栅格矩阵的列数
    im_height = dataset.RasterYSize #栅格矩阵的行数
    im_bands = dataset.RasterCount #波段数
    im_data = dataset.ReadAsArray(0,0,im_width,im_height)#获取数据
    im_geotrans = dataset.GetGeoTransform()#获取仿射矩阵信息
    im_proj = dataset.GetProjection()#获取投影信息
    driver = gdal.GetDriverByName("GTiff")
    return dataset,im_width,im_height,im_bands,im_data,im_geotrans,im_proj,driver

def readShpEnvelop(fileNmae):
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    # 支持中文编码
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    ogr.RegisterAll()
    # 设置driver,并打开矢量文件
    driver = ogr.GetDriverByName('ESRI Shapefile')

    ds = driver.Open(fileNmae, 0)
    if ds is None:
        print("Could not open", 'sites.shp')
    # 获取图册
    layer = ds.GetLayer()
    # 要素数量
    numFeatures = layer.GetFeatureCount(0)
    result_listsll = []
    Envelopelist = []
    # print("Feature count: " + str(numFeatures))
    for i in range(0,numFeatures):
        feature = layer.GetFeature(i)
        id = feature.GetFieldAsString("id")
        if id == '1':
            result_list = []
            # 获取空间属性
            geometry = feature.GetGeometryRef()
            geom = str(geometry)
            geom1 = re.findall(r'[(](.*)[)]',geom)
            geom2 =''.join(geom1)
            geom3 = geom2.split('),')
            for i in range(0,len(geom3)):
                geom4 = geom3[i]
                geomarr = geom4.replace(")","")
                geomarr = geomarr.replace("(","")
                geomarr = geomarr.split(',')
                for k in geomarr:
                    result_list.append((float(k.split(' ')[0]),float(k.split(' ')[1])))
                # try:
                #     result_list = [[float(j) for j in k] for k in result_list]
                # except:
                #  print(fileNmae)

                # result_list = np.array(result_list).reshape(len(result_list), 2)
                result_listsll.append(result_list)
                result_list = []

            MaxX = -200
            MaxY = -200
            MinX = 1000
            MinY = 1000
    # for i in range(0,numFeatures):
    #     feature = layer.GetFeature(i)
    #     # 获取空间属性
    #         geometry = feature.GetGeometryRef()
            polygonextent = geometry.GetEnvelope()

            if MaxX< max(MaxX,max(polygonextent[0],polygonextent[1])):
               MaxX = max(MaxX,max(polygonextent[0],polygonextent[1]))
            if MaxY < max(MaxY,max(polygonextent[2], polygonextent[3])):
               MaxY = max(MaxY,max(polygonextent[2], polygonextent[3]))
            if MinX > min(MinX,min(polygonextent[0],polygonextent[1])):
               MinX = min(MinX,min(polygonextent[0],polygonextent[1]))
            if MinY > min(MinY,min(polygonextent[2], polygonextent[3])):
               MinY = min(MinY,min(polygonextent[2], polygonextent[3]))

            Envelopelist = [(MinX, MaxY), (MaxX, MaxY), (MaxX, MinY), (MinX, MinY)]
        else:
            pass
    result_listsll = sorted(result_listsll,key =lambda x:len(x))[-1]
    return result_listsll, Envelopelist

#遍历shp文件路径，判断该图像和其中哪一个有交叠
def ShpIntersectsTif(dataset_tiff,im_height,im_width,path):
    flag = False
    shp_names = os.listdir(path)
    coords1 = imagexy2geo(dataset_tiff, 0, 0)
    coords1_1 = geo2lonlat(dataset_tiff, coords1[0], coords1[1])
    coords2 = imagexy2geo(dataset_tiff, im_height - 1, 0)
    coords2_1 = geo2lonlat(dataset_tiff, coords2[0], coords2[1])
    coords3 = imagexy2geo(dataset_tiff, im_height - 1, im_width - 1)
    coords3_1 = geo2lonlat(dataset_tiff, coords3[0], coords3[1])
    coords4 = imagexy2geo(dataset_tiff, 0, im_width - 1)
    coords4_1 = geo2lonlat(dataset_tiff, coords4[0], coords4[1])
    coords_tiff = [(coords1_1[0], coords1_1[1]), (coords4_1[0], coords4_1[1]), (coords3_1[0], coords3_1[1]),
                   (coords2_1[0], coords2_1[1])]
    coords_tiff = np.array(coords_tiff).reshape(4, 2)
    a = Polygon(coords_tiff)
    for shp_name in shp_names:
        if shp_name[-4:] == ".shp":
            shplist, Envelopelist = readShpEnvelop(
                os.path.join(path, shp_name))
            coords_shp = np.array(Envelopelist).reshape(4, 2)
            b = Polygon(coords_shp)
            if b.intersects(a):
                flag = True
                break
            else:
                pass
    return shplist,flag
# 图像切片
slice_h = slice_w = 1024
step = 600

def WindowIntersectShp(windows,dataset_tiff,shplist):
    new_widows=[]
    flag = False
    window_flag = []
    for window in windows:
        subimg_coords1 = imagexy2geo(dataset_tiff, window[2], window[3])
        subimg_coords1_1 = geo2lonlat(dataset_tiff, subimg_coords1[0], subimg_coords1[1])
        subimg_coords2 = imagexy2geo(dataset_tiff, window[2]+800, window[3])
        subimg_coords2_1 = geo2lonlat(dataset_tiff, subimg_coords2[0], subimg_coords2[1])
        subimg_coords3 = imagexy2geo(dataset_tiff, window[2]+800, window[3]-800)
        subimg_coords3_1 = geo2lonlat(dataset_tiff, subimg_coords3[0], subimg_coords3[1])
        subimg_coords4 = imagexy2geo(dataset_tiff, window[2], window[3]-800)
        subimg_coords4_1 = geo2lonlat(dataset_tiff, subimg_coords4[0], subimg_coords4[1])

        # subimg_coords_shp = [subimg_coords1_1, subimg_coords2_1, subimg_coords3_1, subimg_coords4_1]
        subimg_coords_shp = [(subimg_coords1_1[0], subimg_coords1_1[1]), (subimg_coords4_1[0], subimg_coords4_1[1]),
                       (subimg_coords3_1[0], subimg_coords3_1[1]), (subimg_coords2_1[0], subimg_coords2_1[1])]
        subimg_coords_shp = np.array(subimg_coords_shp).reshape(4, 2)

        # for n in range(0, len(shplist)):
        #     # coords_shp1 = np.array(shplist[n]).reshape(len(shplist[n]), 2)
        #     coords_shp1 = shplist[n].reshape(len(shplist[n]), 2)
        #     b1 = Polygon(coords_shp1)
        #     if Polygon(subimg_coords_shp).intersects(b1):
        #         new_widows.append(window)
        #     else:
        #         pass
        # for n in range(0, len(shplist)):
            # coords_shp1 = np.array(shplist[n]).reshape(len(shplist[n]), 2)
        # coords_shp1 = shplist[n].reshape(len(shplist[n]), 2)
        b1 = Polygon(shplist)
        if Polygon(subimg_coords_shp).intersects(b1):
            flag = True
            window_flag.append(flag)
            new_widows.append(window)
            flag = False
        else:
            window_flag.append(flag)

    return window_flag,np.array(new_widows)

def Detection2txt(path,classsname,detections,threshold=0.2, classwise=False):
    if classwise:
        for index,class_det in enumerate(detections):
            for det in class_det:
                score = det[-1]
                if score > threshold:
                    with open(path + '.txt', "a") as f:
                        f.write(str(classsname[index]) + ' ' + str(score) + ' ' + str(int(det[0])) + ' ' + str(
                            int(det[1])) + ' ' + str(int(det[2])) + ' ' + str(int(det[3])) + '\n')
                else:
                    continue
    else:
        for det in detections:
            score = det[-2]
            if score > threshold:
                with open(path+'.txt',"a") as f:
                    f.write(str(classsname[int(det[-1])])+' '+str(det[-2])+' '+str(int(det[0]))+' '+str(int(det[1]))+' '+str(int(det[2]))+' '+str(int(det[3]))+'\n')
            else:
                continue