import cv2
import numpy as np
import torch
from ultralytics import YOLO
import math
from scipy.stats import entropy


def opticalflow_edge(video_path, regions, w1, w2, file_path):
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    # 读取第一帧（灰度）
    ret, prev_frame = cap.read()
    if not ret:
        print("无法读取视频")
        cap.release()
        exit()

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    framecount=1
    while cap.isOpened():
        # 读取下一帧
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = dis_flow.calc(prev_gray, gray, None)
        # 存储每个区域的光流标准差
        flag_values = [w1 * np.std(flow[y:y + h, x:x + w].reshape(-1, 2)) for x, y, w, h in regions]

        edges = cv2.Canny(gray, 50, 150)  # 只计算一次边缘图
        # 计算每个区域的边缘像素比例
        for idx, (x, y, w, h) in enumerate(regions):
            region_edges = edges[y:y + h, x:x + w]  # 直接切片
            edge_ratio = np.count_nonzero(region_edges) / (w * h)  # 使用 np.count_nonzero() 代替 np.sum(region_edges > 0)
            flag_values[idx] += w2 * edge_ratio  # 累加权重

        # 将每帧的flag存入文件
        with open(file_path , 'w') as f:
            f.write(str(framecount) + ' ' + (' '.join(map(str, flag_values))) + '\n')
        # 更新前一帧
        prev_gray = gray
        framecount+=1

    cap.release()


if __name__ == '__main__':
    vdo1region = [[280, 0, 350, 200], [150, 200, 550, 220], [0, 420, 735, 300], [920, 282, 360, 179],
                  [756, 461, 425, 259]]
    vdo2region = [[618, 240, 123, 37], [500, 277, 297, 58], [280, 335, 329, 130], [678, 335, 285, 130],
                  [0, 465, 575, 255], [696, 465, 583, 255]]
    vdo3region = [[560, 222, 160, 45], [310, 267, 320, 124], [660, 267, 307, 124], [0, 391, 580, 173],
                  [690, 391, 589, 173], [0, 564, 545, 156], [720, 564, 559, 156]]
    vdo4region = [[668, 0, 259, 102], [520, 102, 560, 195], [350, 297, 880, 195], [150, 492, 1130, 227]]
    vdopath = ["../traffic/vdo1/vdo1.mp4", "../traffic/vdo2/vdo2.mp4", "../traffic/vdo3/vdo3.mp4",
               "../traffic/vdo4/vdo4.mp4"]
    vdoregion = [vdo1region, vdo2region, vdo3region, vdo4region]
    flag_paths = ["../traffic/vdo1/vdo1flag1.txt","../traffic/vdo2/vdo2flag1.txt","../traffic/vdo3/vdo3flag1.txt","../traffic/vdo4/vdo4flag1.txt"]
    w1=1
    w2=1
    opticalflow_edge(vdopath[0], vdoregion[0], w1, w2, flag_paths[0])