from glob import glob
from utils.utils import CFG, Model
import torch.nn.functional as F


import cv2
import numpy as np
import torch
import os

from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine

import time
# import socket
import threading

import pandas as pd
import flyr

def getTemp(thermogram, x1, x2, y1, y2):
    thermal = thermogram.celsius

    return thermal[y1:y2, x1:x2].max()

def prepare():
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


def plot_result(result, imgfp, timgfp, output_path):

    img = imread(imgfp)
    timg = imread(timgfp)

    bboxes = np.vstack(result)
    # thermogram = flyr.unpack(timgfp)
    cnt = 0

    for i in bboxes:
        x1 = round(i[0])
        y1 = round(i[1])
        x2 = round(i[2])
        y2 = round(i[3])
        confidence = i[4]
        
        bbox_result.append(f'face {confidence} {x1} {y1} {x2} {y2}')

        # class probability x1 y1 x2 y2
        copyImg = img[y1:y2, x1:x2].copy()
        copytImg = timg[y1:y2, x1:x2].copy()

        # copytImg = timg[int(x1):int(x2), int(y1-20):int(y2-20)].copy()
        # temp = getTemp(thermogram, x1, x2, y1, y2)
        cv2.imwrite('{}/face_{}_normal'.format(output_path, str(cnt).zfill(3))+'.jpg', copyImg)
        cv2.imwrite('{}/face_{}_thermal'.format(output_path, str(cnt).zfill(3))+'.jpg', copytImg)
        # print(f'temp:{temp}')
        cnt +=1

def pred(input_path, output_path):

    imgname = f"{input_path}/photo.jpg"
    timgname = f"{input_path}/msx.jpg"

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
    plot_result(result, imgname, timgname, output_path)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = Config.fromfile('configs/infer/tinaface/tinaface_r50_fpn_bn.py')
    class_names = cfg.class_names
    engine, data_pipeline, device = prepare()

    model = Model()
    model.load_state_dict(torch.load(CFG.weight_path))
    model.to(CFG.device)

    resultList = ['mask', 'nomask', 'wrong']

    # IP = '192.168.0.170'
    # PORT = 28285
    # SIZE = 1024
    # ADDR = (IP, PORT)

    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    #     server_socket.bind(ADDR)
    #     server_socket.listen()
    #     print('booted')
    with torch.no_grad():
        model.eval()
        while True:
            st = time.time()
            # client_socket, client_addr = server_socket.accept()
            # msg = client_socket.recv(SIZE)
            # path = str(msg.decode())
            path = '../files/device001/2022-06-02_152107.60'
            input_path = f'{path}/input'
            output_path = f'{path}/output'

            # client_socket.sendall("done".encode())
            # client_socket.close()
            img_tensors = []
            pred_result = []
            bbox_result = []
            # pred(input_path, output_path)
            threads = threading.Thread(target=pred, args=(input_path, output_path))  # Multi Thread
            threads.start()
            threads.join()

            image_paths = sorted(glob(output_path+"/*normal*"))

            for image_path in image_paths:
                img = cv2.imread(image_path)
                transformed = CFG.transformed(image=img)
                img = transformed['image'].unsqueeze(0).float().to(CFG.device)
                img_tensors.append(img)

            images = torch.cat(img_tensors, dim=0)
            outputs = model(images)
            threshold = F.softmax(outputs, dim=1)
            print(f'threshold:{threshold}')
            print(torch.max(threshold, dim=1).values)

            maxIndex = torch.argmax(outputs, dim=1).tolist()

            # 0 mask 1 nomask 2 wrong
            for idx in maxIndex:
                pred_result.append(resultList[idx])

            # save result image_paths and visualization
            result_data = {'image_path': image_paths, 'pred_result': pred_result, 'bbox_result': bbox_result}
            df = pd.DataFrame(data=result_data)
            df.to_csv(output_path+'/pred.csv', index=False)

            # print(resultList[maxIndex])
            # print(outputs.size())

            ed = time.time()
            print(f'{ed-st}s passed')

