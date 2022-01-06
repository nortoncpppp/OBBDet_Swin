from argparse import ArgumentParser

from mmdet.apis import init_detector, show_result_pyplot
from mmdet.apis import inference_detector_huge_image,inference_detector_huge_image1
import cv2,os
import numpy as np
import mmcv
from demo.function import readTif,ShpIntersectsTif,Detection2txt
import time
#
plane40_colormap = [
    (54, 67, 244),
    (99, 30, 233),
    (176, 39, 156),
    (183, 58, 103),
    (181, 81, 63),
    (243, 150, 33),
    (212, 188, 0),
    (136, 150, 0),
    (80, 175, 76),
    (74, 195, 139),
    (57, 220, 205),
    (59, 235, 255),
    (0, 152, 255),
    (34, 87, 255),
    (72, 85, 121),
    (54, 67, 244),
    (99, 30, 233),
    (176, 39, 156),
    (183, 58, 103),
    (181, 81, 63),
    (243, 150, 33),
    (212, 188, 0),
    (136, 150, 0),
    (80, 175, 76),
    (74, 195, 139),
    (57, 220, 205),
    (59, 235, 255),
    (0, 152, 255),
    (34, 87, 255),
    (72, 85, 121),
    (243, 150, 33),
    (212, 188, 0),
    (136, 150, 0),
    (80, 175, 76),
    (74, 195, 139),
    (57, 220, 205),
    (59, 235, 255),
    (0, 152, 255),
    (34, 87, 255),
    (72, 85, 121)
]
classnames = ['HongZJ-B1','HongZJ-B52','HongZJ-VT27','JiaoLJ-VT31','MinYKJ',
           'QTJJ','WuRJ-RQ4','YuJJ-E2','YuJJ-E3','YuJJ-E4',
           'YuJJ-E6','YuJJ-P3','YuJJ-P8','YunSJ-An26','YunSJ-C130',
           'YunSJ-C17','YunSJ-C2','YunSJ-C-27J','YunSJ-C5','YunSJ-IL76',
           'YunSJ-KC10','YunSJ-KC135','YunSJ-KC767','YunSJ-L1011','YunSJ-QT',
           'ZhanDJ-A10','ZhanDJ-F15','ZhanDJ-F16','ZhanDJ-F18','ZhanDJ-F22',
           'ZhanDJ-F35','ZhanDJ-F5','ZhanDJ-HY2000','ZhanDJ-QT','ZhenCJ-SHADOW-R1',
           'ZhenCJ-U2','ZhiSJ-CH47','ZhiSJ-CH53','ZhiSJ-QT','ZhiSJ-V22']
def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 10)):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    return img

def draw_poly_detections_pic(img, detections, class_names, scale, threshold=0.2, putText=False,showStart=False, colormap=None):
    """

    :param img:
    :param detections:
    :param class_names:
    :param scale:
    :param cfg:
    :param threshold:
    :return:
    """
    import pdb
    import cv2
    import random
    assert isinstance(class_names, (tuple, list))
    img = mmcv.imread(img)
    color_white = (255, 255, 255)

    for det in detections:
        bbox = det[:4] * scale
        score = det[-2]
        class_index = (int)(det[-1])
        if colormap is None:
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        else:
            color = colormap[class_index]

        if score < threshold:
            continue
        bbox = list(map(int, bbox))
        # if showStart:
        #     cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
        # for i in range(3):
        #     cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]), color=color, thickness=2,lineType=cv2.LINE_AA)
        # cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2,lineType=cv2.LINE_AA)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
        if putText:
            cv2.putText(img, '%s %.3f' % (class_names[class_index], score), (bbox[0], bbox[1] + 10),
                        color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return img

def main():
    start_time = time.time()
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        'split', help='split configs in BboxToolkit/tools/split_configs')
    # 数据集路径
    parser.add_argument("--input_dir", default='/media/think/D8EA260719113F00/0data/GaoJing/ps/lmy/geo', help="input path", type=str)
    # 机场shp路径
    parser.add_argument("--input_shp_dir", default='/media/think/新加卷/1code/GetThumbImage/GetThumbImage/Mask/10_shp/', help="input shp path", type=str)
    parser.add_argument("--output_dir", default='/media/think/D8EA260719113F00/0data/GaoJing/ps/lmy/geo/out/', help="output path",
                        type=str)
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    nms_cfg = dict(type='BT_nms', iou_thr=0.1)

    # result,result_pic = inference_detector_huge_image(
    #     model, args.img, args.split, nms_cfg)
    # # show the results
    # # print(result)
    # # img = show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
    # # cv2.imwrite("res.tif", img)
    # img = draw_poly_detections_pic(args.img, result_pic, classnames, scale=1, threshold=0.2,
    #                                colormap=plane40_colormap, putText=True)
    # cv2.imwrite("res.tif", img)

    img_names = os.listdir(args.input_dir)
    for img_name in img_names:
        if img_name[-4:] == ".tif":
            dataset_tiff, im_width, im_height, im_bands, im_data, im_geotrans, im_proj, driver = readTif(
                os.path.join(args.input_dir, img_name))
            shplist,flag = ShpIntersectsTif(dataset_tiff,im_height,im_width,args.input_shp_dir)
            if flag ==True:
                print('intersect with shp!')
                result, result_pic = inference_detector_huge_image1(
                    model, os.path.join(args.input_dir, img_name), args.split, nms_cfg,shplist,dataset_tiff)
            else:
                result, result_pic = inference_detector_huge_image(
                    model, os.path.join(args.input_dir, img_name), args.split, nms_cfg)

            # img = draw_poly_detections_pic(os.path.join(args.input_dir, img_name), result_pic, classnames, scale=1, threshold=0.2,
            #                                colormap=plane40_colormap, putText=True)
            # cv2.imwrite(os.path.join(args.output_dir, img_name), img)
            Detection2txt(os.path.join(args.input_dir, img_name[:-4]),classnames,result_pic,threshold=0.2,classwise=False)
            # img = show_result_pyplot(model, os.path.join(args.input_dir, img_name), result, score_thr=args.score_thr)
            # cv2.imwrite("res.tif", img)

            print(img_name,'process time:', time.time() - start_time)

    print('total time:', time.time() - start_time)


if __name__ == '__main__':
    main()
