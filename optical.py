import cv2
import numpy as np
import torch
from ultralytics import YOLO
import math
from scipy.stats import entropy

def non_max_suppression(boxes, sortby="conf" ,iou_threshold=0.2):
    if len(boxes) == 0:
        return []

    # 提取所有框的 [x1, y1, x2, y2] 和置信度 conf
    coords = np.array([box[0] for box in boxes])  # [x1, y1, x2, y2]
    scores = np.array([box[2] for box in boxes])  # conf

    # 计算每个框的面积
    areas = (coords[:, 2] - coords[:, 0] + 1) * (coords[:, 3] - coords[:, 1] + 1)
    if sortby == "conf":
        # 按置信度降序排序
        order = scores.argsort()[::-1]
    elif sortby == "area":
        # 按面积大小降序排序
        order = areas.argsort()[::-1]

    keep = []  # 保存保留下来的框索引

    while order.size > 0:
        i = order[0]  # 当前置信度最高的框索引
        keep.append(i)

        # 计算当前框与剩余框的交并比(IOU)
        xx1 = np.maximum(coords[i, 0], coords[order[1:], 0])  # 左上角x的最大值
        yy1 = np.maximum(coords[i, 1], coords[order[1:], 1])  # 左上角y的最大值
        xx2 = np.minimum(coords[i, 2], coords[order[1:], 2])  # 右下角x的最小值
        yy2 = np.minimum(coords[i, 3], coords[order[1:], 3])  # 右下角y的最小值

        # 计算交集面积
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        # 计算交并比(IOU)
        #iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留IOU小于阈值的框
        #inds = np.where(iou <= iou_threshold)[0]
        inds = np.where(inter ==0)[0]
        order = order[inds + 1]

    # 根据保留下来的索引返回框
    return [boxes[i] for i in keep]

def Isincluded(box, track_regions):
    if not track_regions:
        return False
    x1, y1, x2, y2 = box
    for region in track_regions:
        r_x1, r_y1, r_x2, r_y2 = region
        # 检查 box 是否被 roi 完全包含
        if x1 >= r_x1 and y1 >= r_y1 and x2 <= r_x2 and y2 <= r_y2:
            return True
    return False

def compute_iou(box1, box2):
    """
    计算两个框的IoU
    box1, box2: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算各自的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算IoU
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0

    return iou

def compute_intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    return inter_area

def compute_distance(box1, box2):
    """计算两个框之间的最小边界距离"""
    left = box2[0] - box1[2]  # box2 左边到 box1 右边的距离
    right = box1[0] - box2[2]  # box1 左边到 box2 右边的距离
    top = box2[1] - box1[3]  # box2 顶边到 box1 底边的距离
    bottom = box1[1] - box2[3]  # box1 顶边到 box2 底边的距离

    horizontal_overlap = max(left, right, 0)
    vertical_overlap = max(top, bottom, 0)
    return max(horizontal_overlap, vertical_overlap,0)

def are_boxes_mergeable(box1, box2, horizontal_threshold=20, vertical_threshold=5):
    # 检查左右边界是否接近
    left_diff = abs(box1[0] - box2[0])
    right_diff = abs(box1[2] - box2[2])
    if left_diff > horizontal_threshold or right_diff > horizontal_threshold:
        return False  # 左右边界差异过大
    # 检查垂直距离（考虑两种情况）
    vertical_diff1 = box2[1] - box1[3]  # 框1在上方，框2在下方
    vertical_diff2 = box1[1] - box2[3]  # 框2在上方，框1在下方

    if (0 <= vertical_diff1 <= vertical_threshold) or (0 <= vertical_diff2 <= vertical_threshold):
        return True  # 满足垂直距离条件

    return False

def merge_two_boxes(box1, box2):#[x1, y1, x2, y2], [cx, cy, w, h], conf, cls
    """合并两个框，返回最小外接矩形"""
    x1 = min(box1[0][0], box2[0][0])
    y1 = min(box1[0][1], box2[0][1])
    x2 = max(box1[0][2], box2[0][2])
    y2 = max(box1[0][3], box2[0][3])
    cx=int((x1+x2)/2)
    cy=int((y1+y2)/2)
    w=x2-x1
    h=y2-y1
    if box1[1][2]*box1[1][3]>box2[1][2]*box2[1][3]:
        conf=box1[2]
        cls=box1[3]
    else:
        conf=box2[2]
        cls=box2[3]
    return [[x1, y1, x2, y2],[cx, cy, w, h], conf, cls]

def merge_boxes(boxes, distance_threshold=10):
    # 合并逻辑
    merged_boxes = []

    while boxes:
        current_box = boxes.pop(0)
        merged = False

        for i in range(len(merged_boxes)):
            existing_box = merged_boxes[i]
            #distance = compute_distance(current_box[0], existing_box[0])
            # 距离小于距离阈值，则合并
            #if distance <distance_threshold:
            if are_boxes_mergeable(current_box[0], existing_box[0]):
                merged_boxes[i] = merge_two_boxes(current_box, existing_box)
                merged = True
                break
        if not merged:
            merged_boxes.append(current_box)

    return merged_boxes

def filter_boxes(track_boxes, track_regions):
    filtered_boxes = []
    for box in track_boxes:
        [x1, y1, x2, y2],[cx,cy,w,h] ,conf, cls = box
        for region in track_regions:
            rx1, ry1, rx2, ry2 = region
            if rx1<=cx<=rx2 and ry1<=cy<=ry2:#中心点在这个区域
                filtered_boxes.append(box)
            """# 计算框和区域的交集
            intersect_x1 = max(x1, rx1)
            intersect_y1 = max(y1, ry1)
            intersect_x2 = min(x2, rx2)
            intersect_y2 = min(y2, ry2)
            # 检查交集是否为有效框
            if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                # 添加交集框到结果列表
                cx=int((intersect_x1+intersect_x2)/2)
                cy=int((intersect_y1+intersect_y2)/2)
                w=intersect_x2-intersect_x1
                h=intersect_y2-intersect_y1
                filtered_boxes.append([[intersect_x1, intersect_y1, intersect_x2, intersect_y2],[cx,cy,w,h] ,conf, cls])
                """
    return filtered_boxes

def yolov8n_roi(frame,roi):#roi位置坐标[x1,y1,x2,y2]
    # 初始化YOLOv8模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO('../model/yolov8n.pt').to(device)
    roi_img = frame[roi[1]:roi[3], roi[0]:roi[2]]
    w = roi[2] - roi[0]
    h = roi[3] - roi[1]
    size = 32 * math.ceil(max(w, h) / 32.0)
    # 设置输入图像大小
    model.overrides["imgsz"] = size  # 必须是32的整数倍
    results = model(roi_img)#, conf=0.4
    # 获取边界框信息
    boxes = results[0].boxes  # 批处理的原因
    xyxy = boxes.xyxy.cpu().numpy().astype(int)  # 获取边界框坐标
    xywh = boxes.xywh.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    detect_class=[0,1,2,3,5,7]
    detections = [[a, b, c, d] for a, b, c, d in zip(xyxy, xywh, conf, cls) if d in detect_class]
    for detection in detections:
        # 坐标变换
        detection[0][0] += roi[0]
        detection[0][1] += roi[1]
        detection[0][2] += roi[0]
        detection[0][3] += roi[1]
        detection[1][0] += roi[0]
        detection[1][1] += roi[1]
    return detections

def yolov8m_frame(frame):
    # 初始化YOLOv8模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO('../model/yolov8m.pt').to(device)
    size = 1280
    # 设置输入图像大小
    model.overrides["imgsz"] = size  # 必须是32的整数倍
    results = model(frame)
    # 获取边界框信息
    boxes = results[0].boxes  # 批处理的原因
    xyxy = boxes.xyxy.cpu().numpy().astype(int)  # 获取边界框坐标
    xywh = boxes.xywh.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    detect_class = [0, 1, 2, 3, 5, 7]
    detections = [[a, b, c, d] for a, b, c, d in zip(xyxy, xywh, conf, cls) if d in detect_class]
    return detections

def calculate_entropy(image):
    """计算图像的熵"""
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    hist_normalized = hist / np.sum(hist)
    return entropy(hist_normalized, base=2)

def yolov8_opticalflow_new(camera_path,regions,region_areas,width,height,fps,label_path):
    # 打开视频文件
    cap = cv2.VideoCapture(camera_path)  # 或者使用0来打开摄像头

    # 初始化变量
    prev_gray = None
    prev_boxes = []#[x1, y1, x2, y2],[cx,cy,w,h] , conf, cls = box,维护这个列表展示每帧的检测结果
    p0=None
    frame_count = 0  # 用于计数帧数
    prev_flag_values=[]
    region_detect=[0 for _ in regions]#整个视频每个区域检测次数
    # 创建随机生成的颜色
    color = np.random.randint(0, 255, (100, 3))
    flag_path = camera_path.replace(".mp4", "flag1.txt")
    detect_path=camera_path.replace(".mp4", "detect.txt")
    with open(detect_path, "w") as f1:#清除内容
        pass
    with open(flag_path, "w") as f:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if frame_count % fps == 0:
                detections=yolov8m_frame(frame)
                # 绘制检测到的边界框与点
                for i,box in enumerate(detections):
                    [x1, y1, x2, y2],[cx,cy,w,h] ,conf, cls = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), 5, color[i % len(color)].tolist(), -1)
                # 更新
                prev_flag_values = []
                prev_gray = gray
                prev_boxes = detections
                p0 = np.array([((box[0][0] + box[0][2]) / 2, (box[0][1] + box[0][3]) / 2) for box in prev_boxes],dtype=np.float32)

            else:
                flag_values = [0 for _ in regions]
                track_boxes = []  # 记录跟踪成功的点,然后和检测结果做匹配
                if len(p0)!=0:
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
                    for i, (old, new) in enumerate(zip(p0, p1)):
                        if st[i] == 1:
                            a, b = new.ravel().astype(int)
                            w = prev_boxes[i][1][2]
                            h = prev_boxes[i][1][3]
                            # 更新位置
                            prev_boxes[i][1][0] = a
                            prev_boxes[i][1][1] = b
                            prev_boxes[i][0][0] = max(0, int(a - w / 2))
                            prev_boxes[i][0][1] = max(0, int(b - h / 2))
                            prev_boxes[i][0][2] = min(width, int(a + w / 2))
                            prev_boxes[i][0][3] = min(height, int(b + h / 2))
                            #prev_boxes[i][2]=0#conf=0
                            track_boxes.append(prev_boxes[i])
                    # 光流标准差
                    w1 = 1
                    if np.sum(st) > 0:
                        # 将 st 转换为布尔型数组，表示哪些特征点成功跟踪
                        st1 = st.reshape(-1)  # 变为一维数组
                        # 只取成功跟踪的特征点（成功的特征点在 st == 1 的地方）
                        flow = p1[st1 == 1]
                        for idx, (x, y, w, h) in enumerate(regions):
                            # 筛选在当前区域内的特征点
                            in_region = np.array([True if x <= fx < x + w and y <= fy < y + h else False for fx, fy in flow])
                            region_flow_in_region = flow[in_region]
                            N = len(region_flow_in_region)
                            # 如果当前区域有光流数据
                            if N > 0:
                                magnitudes = np.linalg.norm(region_flow_in_region - p0[st1 == 1][in_region], axis=1)
                                std_dev = np.std(magnitudes)
                                flag_values[idx] += w1 * std_dev
                                #print(f"Region {idx + 1} std_dev: {std_dev:.2f}")
                            else:
                                flag_values[idx] += 0
                w2 = 1
                edges = cv2.Canny(gray, 50, 150)
                for idx, (x, y, w, h) in enumerate(regions):
                    region_edges = edges[y:y + h, x:x + w]
                    r_area=w*h
                    edge_pixel_count = np.sum(region_edges > 0)  # 边缘像素数量（非零像素计数）
                    edge_ratio = edge_pixel_count / r_area  # 计算比例
                    flag_values[idx] += w2 * edge_ratio
                    #print(f"Region {idx + 1} edge_ratio: {edge_ratio:.2f}")


                # 把每个区域的指标写入文件 vdo1flag1.txt,检测的区域放在 rois
                track_regions = []
                detections = []
                sum_area=0
                for idx, (x, y, w, h) in enumerate(regions):
                    y2=y+h
                    padding_ratio1=(y/height)*0.5
                    padding_ratio2=(y2/height)*0.5

                    new_y1=max(0,int(y-padding_ratio1*h))
                    new_y2=min(height,int(y2+padding_ratio2*h))

                    roi = [x, new_y1, x+w, new_y2]
                    """detect_w = 32 * math.ceil((roi[2] - roi[0]) / 32)
                    detect_h = 32 * math.ceil((roi[3] - roi[1]) / 32)
                    detect_area = detect_h * detect_w"""
                    f.write(str(frame_count)+" "+str(idx)+" "+str(flag_values[idx])+"\n")
                    # flag_values[idx] >1.5  or 0.3 3
                    if len(prev_flag_values) != 0 and flag_values[idx] - prev_flag_values[idx] > 0.1*prev_flag_values[idx]:  # 对这个区域检测
                        sum_area += region_areas[idx]
                        region_detect[idx]+=1
                        region_results=yolov8n_roi(frame, roi)
                        #if len(region_results)==0:
                            #cv2.imwrite(f"../traffic/vdo2/results/frame{frame_count}region{idx}.jpg", frame[roi[1]:roi[3], roi[0]:roi[2]])
                        detections.extend(region_results)
                    else:
                        track_regions.append([x, y, x+w,y+h])
                with open(detect_path,"a") as f1:
                    f1.write(str(frame_count)+" "+str(sum_area)+"\n")

                detections = non_max_suppression(detections,"area")#先按照面积合并
                track_boxes=filter_boxes(track_boxes,track_regions)#只保留交集到达一定程度的框
                detections.extend(track_boxes)
                detections = non_max_suppression(detections,"area")
                prev_flag_values = flag_values
                prev_gray = gray
                prev_boxes = detections
                p0 = np.array([[(box[0][0] + box[0][2]) / 2, (box[0][1] + box[0][3]) / 2] for box in prev_boxes],dtype=np.float32)

                for i, box in enumerate(prev_boxes):
                    [x1, y1, x2, y2], [cx, cy, w, h], conf, cls = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.circle(frame, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 5, color[i % len(color)].tolist(), -1)
                """if frame_count==33 or frame_count==50:
                    cv2.imwrite("../traffic/vdo1/results/frame%d.jpg" % frame_count,frame)"""
            # 每帧的检测框写入文件
            with open(label_path + "frame" + str(frame_count) + ".txt",
                      'w') as f2:  # [x1, y1, x2, y2],[cx,cy,w,h] , conf, cls = box
                for box in prev_boxes:
                    f2.write(str(box[3]) + ' ' + str(box[2]) + ' ' + (
                        ' '.join(map(str, box[1]))) + '\n')  # cls,conf,cx,cy,w,h
            frame_count += 1  # 增加帧计数器
            # 显示结果
            #cv2.imshow('YOLOv8 with Optical Flow', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    return region_detect

if __name__ == '__main__':
    fps=10
    lk_params = dict(winSize=(21, 21), maxLevel=3,  # 增加窗口大小和金字塔层数，适应快速运动
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    color = np.random.randint(0, 255, (100, 3))
    """vdo1region=[[360, 0, 1207, 529],[0 ,529 ,1920 ,551]]
    vdo2region=[[610 ,0 ,889 ,353],[0, 353, 1920, 727]]
    vdo3region = [[312 ,0, 1351, 478],[0, 478, 1920, 602]]"""
    #vdo1region=[[340, 0, 188, 117],[312 ,117, 310, 100],[150, 217, 550, 203],[0, 420, 735 ,300],[920, 282, 360, 179],[756, 461, 425, 259]]
    vdo1region = [[280, 0, 350, 200],[150, 200, 550, 220], [0, 420, 735, 300],[920, 282, 360, 179], [756, 461, 425, 259]]
    vdo1region_area=[90112,184320,306176,110592,157696]
    vdo2region=[[618, 240 ,123, 37],[500, 277, 297, 58],[280, 335 ,329 ,130],[678 ,335, 285, 130],[0, 465, 575, 255],[696, 465, 583, 255]]
    vdo2region_area=[8192,30720,78848,64512,202752,214016]
    vdo3region=[[560,222, 160, 45],[310 ,267, 320, 124],[660, 267 ,307, 124],[0, 391, 580, 173],[690, 391, 589 ,173],[0,564, 545, 156],[720, 564, 559, 156]]
    vdo3region_area=[10240,61440,61440,175104,175104,129024,129024]
    vdo4region=[[668 ,0, 259, 102],[520, 102, 560, 195],[350, 297, 880, 195],[150, 492, 1130, 227]]
    vdo4region_area=[36864,147456,286720,368640]
    vdopath=["../traffic/vdo1/vdo1.mp4","../traffic/vdo2/vdo2.mp4","../traffic/vdo3/vdo3.mp4","../traffic/vdo4/vdo4.mp4"]
    vdoregion=[vdo1region,vdo2region,vdo3region,vdo4region]
    vdoregion_area=[vdo1region_area,vdo2region_area,vdo3region_area,vdo4region_area]
    #cam1.mp4 1280,960  vdo4 1280*720 30fps
    #yolov8_opticalflow_region("../traffic/vdo1/vdo1.mp4", "../experiment/vdo1label/",1920 , 1080,vdo1region,10)
    """for i in range(4):
        region_detect=yolov8_opticalflow_new(vdopath[i],  vdoregion[i],vdoregion_area[i],1280,720,25)"""
    region_detect = yolov8_opticalflow_new(vdopath[0], vdoregion[0], vdoregion_area[0], 1280, 720, 25,"../experiment/vdo1labely/")
        #print(region_detect)
