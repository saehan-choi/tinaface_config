import argparse

import cv2
import numpy as np
import torch
import os

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine

import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    parser.add_argument('config', help='config file path')
    # parser.add_argument('imgname', help='image file name')

    args = parser.parse_args()
    return args


def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f'GPU : {torch.cuda.get_device_name(0)} is using')
    else:
        device = 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device


def plot_result(result, imgfp, class_names, outfp='out.jpg', output_label_name=False, start=1):
    
    output_label_name = output_label_name
    font_scale = 0.5
    bbox_color = 'green'
    text_color = 'green'
    thickness = 1

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = imread(imgfp)
    width, height = img.shape[:2]
    
    bboxes = np.vstack(result)

    blue = (255, 0, 0)
    font =  cv2.FONT_HERSHEY_PLAIN

    print(bboxes)
    f = open(f"./inference_label_data/{output_label_name}", 'w')
    # class probability x1 y1 x2 y2
    for i in bboxes:
        x1 = i[0]
        y1 = i[1]
        x2 = i[2]
        y2 = i[3]
        confidence = i[4]
        f.write(f'0 {confidence} {round(x1)} {round(y1)} {round(x2)} {round(y2)}\n')
    f.close()


    

    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]

    # !!!!!!!!!!!bounding box!!!!!!!!!!!!
    # 이부분 지금 필요없을거같아서 (지금 우분투 상태가 아니라서 어차피 plot 못띄움) 주석처리 할게요.
    
    labels = np.concatenate(labels)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    ed = time.time()
    print(f'{ed-start}s passed')
    FPS = 1 / (ed-start)

    cv2.putText(img, f'FPS : {round(FPS, 2)}', (width-260, 40), font, 3, blue, 3, cv2.LINE_AA)

    imwrite(img, outfp)
    


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_args()

    # path = './inference_data/inferdata.jpg'
    cfg = Config.fromfile(args.config)

    class_names = cfg.class_names
    # 이거 잘몰ㄹj
    engine, data_pipeline, device = prepare(cfg)
    # infer_data_location = os.listdir(path)

    st = time.time()

    imgname = "./inference_data/inferdata.jpg"

    data = dict(img_info=dict(filename=imgname), img_prefix=None)

    data = data_pipeline(data)

    data = collate([data], samples_per_gpu=1)

    if device != 'cpu':
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        #c just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data
        data['img'] = data['img'][0].data
    
    result = engine.infer(data['img'], data['img_metas'])[0]

    plot_result(result, imgname, class_names, outfp=f'./IMG_FPS_IMAGE/Tinaface FPS_pred.jpg', start = st)


if __name__ == '__main__':
    main()

