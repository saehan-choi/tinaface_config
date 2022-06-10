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
import socket
import threading


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


def plot_result(result, imgfp, timgfp):
    img = imread(imgfp)
    timg = imread(timgfp)

    bboxes = np.vstack(result)
    f = open(f"{output_path}/bbox.txt", 'w')
    cnt = 0
    

    for i in bboxes:
        y1 = i[0]
        x1 = i[1]
        y2 = i[2]
        x2 = i[3]
        confidence = i[4]
        f.write(f'0 {confidence} {round(x1)} {round(y1)} {round(x2)} {round(y2)}\n')
        # class probability x1 y1 x2 y2
        copyImg = img[int(x1):int(x2), int(y1):int(y2)].copy()
        copytImg = timg[int(x1):int(x2), int(y1):int(y2)].copy()
        cv2.imwrite(f'{output_path}/face_{str(cnt).zfill(3)}_normal.jpg', copyImg)
        print(f'{output_path}/face_{str(cnt).zfill(3)}_normal.jpg saved')
        cv2.imwrite(f'{output_path}/face_{str(cnt).zfill(3)}_thermal.jpg', copytImg)
        print(f'{output_path}/face_{str(cnt).zfill(3)}_thermal.jpg saved')
        cnt +=1
    f.close()

def pred():

    st = time.time()

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
    plot_result(result, imgname, timgname)

    ed = time.time()
    print(f'{ed-st}s passed')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cfg = Config.fromfile('configs/infer/tinaface/tinaface_r50_fpn_bn.py')
    class_names = cfg.class_names
    engine, data_pipeline, device = prepare()
    IP = '192.168.0.170'
    PORT = 28285
    SIZE = 1024
    ADDR = (IP, PORT)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(ADDR)
        server_socket.listen()
        print('booted')
        while True:
            client_socket, client_addr = server_socket.accept()
            msg = client_socket.recv(SIZE)
            path = str(msg.decode())
            print(f'\"{path}\"')
            input_path = f'{path}/input'
            output_path = f'{path}/output'
            client_socket.sendall("done".encode())
            client_socket.close()
            t = threading.Thread(target=pred, args=())  # Multi Thread
            t.start()

