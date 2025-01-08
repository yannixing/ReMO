import argparse
import sys
import numpy as np
import copy
import time
import random
import re
import pandas as pd
import math
from matplotlib import pyplot as plt

def get_Region(filelist,Region_areas,alltime,fps):
    Region_list_all_i=[]

    for i,file in enumerate(filelist):#每个设备
        R=len(Region_areas[i])
        Region_list_i = [{} for _ in range(R)]
        with open(file, 'r') as f:
            for line in f:
                items = line.strip().split()  # 按空格分隔
                frame=int(items[0])
                region_idx=int(items[1])
                flag=float(items[2])
                frame=frame%fps
                if frame not in  Region_list_i[region_idx]:
                    Region_list_i[region_idx][frame]=[]#键对应的值为一个列表
                Region_list_i[region_idx][frame].append(flag)
        Region_list_all_i.append(Region_list_i)

    # 按时隙
    Region_all_file = "../traffic/Region_all_file.txt"
    with open(Region_all_file, 'w') as f:
        pass

    Region_list_all_t=[[]for _ in range(alltime)]
    for t in range(alltime):
        for i,Region_list_i in enumerate(Region_list_all_i):#每个设备
            R=len(Region_areas[i])
            Region_list_t_i=[{} for _ in range(R)]
            for r in range(R):
                Region_list_t_i[r]={key: value[t] for key, value in Region_list_i[r].items() if t < len(value)}
            Region_list_all_t[t].append(Region_list_t_i)

    with open(Region_all_file, 'w') as f:
        f.write(str(Region_list_all_t))

    return Region_list_all_t#Region_list_all存储flag

def get_bandwidth(file,N,alltime,b_ratio):
    data = pd.read_csv(file, sep='\t', header=None, usecols=[1], names=['bandwidth'])
    # 转换为列表
    bandwidth_list = data['bandwidth'].tolist()
    needlen=N*N*alltime
    grouped_bandwidth = [bandwidth_list[i:i+alltime] for i in range(0, needlen, alltime)]

    bandwidth_matrix_all = []
    for t in range(alltime):
        count = 0
        bandwidth_matrix = np.zeros(((N + 1), (N + 1)))
        for i in range(1,N+1):
            for j in range(N+1):
                if i!=j:#n*n个
                    bandwidth_matrix[i, j] =grouped_bandwidth[count][t] * b_ratio * (1 + j)
                    count=count+1
        bandwidth_matrix_all.append(bandwidth_matrix)
    return bandwidth_matrix_all

def S_i_M(i,M):#每个Region依次计算相加#yolov8m,yolov8s,yolov8n,yolov5nu
    if i==0:
        return 0
    elif i==1:
        return (0.000271 * M + 36.999027)
    elif i==2:
        return ( 0.000123 * M + 24.681274)#ms(0.00016604785738005167 * M + 37.051006279182076)
    elif i==3:
        return (0.000047 * M + 30.123385)#ms
    elif i==4:
        return (0.000024 * M + 48.337254)

def get_Q_list(last_T_list,Qmin,Qmax):
    Q_list=[]
    k=1
    for i,t in enumerate(last_T_list):
        Q=2 / (1+np.exp(-k * t))-1
        Q_list.append(Q * (Qmax - Qmin) + Qmin)#Q 0.1，1
    return Q_list

def get_Q(t,Qmin,Qmax):
    k = 1
    Q = 2 / (1 + np.exp(-k * t)) - 1
    Q=(Q * (Qmax-Qmin) + Qmin)  # Q 0.1，1.2
    return Q

def initialize_X_matrix_list1(N,fps,Region_list,Q_t):#第一个时隙要执行的初始化算法
    Q_matrix_list = []
    # 得到Q
    for i in range(N):
        R = len(Region_list[i])
        Q_matrix = np.full((fps, R), 0)
        for r, region_dict in enumerate(Region_list[i]):
            pre_flag = -1
            for f, flag in region_dict.items():
                if pre_flag != -1:
                    Q = abs(flag - pre_flag) / pre_flag  # 检测的重要性程度
                    Q_matrix[f, r] = Q
                pre_flag = flag
        Q_matrix_list.append(Q_matrix)

    X_matrix_list = []
    # 每个设备尽量在本地处理
    for i in range(N):
        R = len(Region_list[i])
        X_matrix = np.full((fps, R), -1)
        Q_matrix=Q_matrix_list[i]
        for f in range(fps):
            for r in range(R):
                if Q_matrix[f,r] > Q_t:
                    X_matrix[f, r] = i + 1  # 在本地检测
        # 每区域每时隙至少要检测一次
        for r in range(R):
            if np.all(X_matrix[:, r] == -1):
                f = int(fps / 2)
                X_matrix[f, r] = i + 1
        X_matrix_list.append(X_matrix)

    return X_matrix_list,Q_matrix_list

def initialize_X_matrix_list2(N,fps,Region_list,Q_list):#第一个时隙要执行的初始化算法
    Q_matrix_list = []
    # 得到Q
    for i in range(N):
        R = len(Region_list[i])
        Q_matrix = np.full((fps, R), 0)
        for r, region_dict in enumerate(Region_list[i]):
            pre_flag = -1
            for f, flag in region_dict.items():
                if pre_flag != -1:
                    Q = abs(flag - pre_flag) / pre_flag  # 检测的重要性程度
                    Q_matrix[f, r] = Q
                pre_flag = flag
        Q_matrix_list.append(Q_matrix)

    X_matrix_list = []
    # 每个设备尽量在本地处理
    for i in range(N):
        R = len(Region_list[i])
        X_matrix = np.full((fps, R), -1)
        for r, region_dict in enumerate(Region_list[i]):
            Q = Q_list[i]
            pre_flag = -1
            for f, flag in region_dict.items():
                if pre_flag == -1:
                    pre_flag = flag
                elif flag - pre_flag > Q * pre_flag:  # 需要重检测
                    X_matrix[f, r] = i + 1  # 在本地检测
                    pre_flag = flag
        # 每区域每时隙至少要检测一次
        for r in range(R):
            if np.all(X_matrix[:, r] == -1):
                f = int(fps / 2)
                X_matrix[f, r] = i + 1
        X_matrix_list.append(X_matrix)

    return X_matrix_list,Q_matrix_list

def N_average_Accuracy(N,accuracy_list,fps,X_matrix_list,Q_matrix_list):
    A_N=[]
    for i in range(N):
        X_matrix=X_matrix_list[i]
        Q_matrix=Q_matrix_list[i]
        A_i = [accuracy_list[i][0]]#设备i上的视频每帧的精度
        #每个区域精度取平均
        R=X_matrix.shape[1]
        pre_A_region=[accuracy_list[i][0] for r in range(R)]
        for f in range(1,fps):
            A_region=[ 0 for _ in range(R)]
            for r in range(R):
                if X_matrix[f,r]!=-1:
                    A_region[r]=accuracy_list[i][X_matrix[f,r]]#检测精度
                else:
                    A_region[r] = pre_A_region[r]*min(np.exp(-Q_matrix[f,r]),phi)#跟踪精度
            pre_A_region=A_region
            A_i.append(sum(A_region)/R)
        average_A_i=sum(A_i)/len(A_i)
        A_N.append(average_A_i)
    return sum(A_N)/len(A_N)#返回所有设备平均精度的平均精度

def N_inference_Time(N,fps,X_matrix_list,Region_areas):#Region_areas存储面积,每设备，每区域的面积列表
    T_inference=[0 for _ in range(N)]
    for i in range(N):#遍历所有设备
        X_matrix = X_matrix_list[i]
        i_areas = Region_areas[i]
        for f in range(1, fps):
            for r in range(X_matrix.shape[1]):
               if X_matrix[f, r] !=-1:
                    j=X_matrix[f, r]
                    T_inference[j-1]+=S_i_M(j,i_areas[r])*0.001#s
    return T_inference

def N_transmission_Time(N,m0,arfa,fps,X_matrix_list,Region_areas,bandwidth_matrix):#bandwidth_matrix为(N+1)*（N+1）矩阵，当i=j时为空
    T_send=[0 for _ in range(N)]#卸载的帧
    T_receive = [0 for _ in range(N)]#接收的帧
    for i in range(N):
        X_matrix=X_matrix_list[i]
        i_areas=Region_areas[i]
        T_send[i]+=arfa*m0/(bandwidth_matrix[i+1,0]*8*pow(2,20))#MB/s
        for f in range(1,fps):
            for r in range(X_matrix.shape[1]):
                j=X_matrix[f, r]-1
                if j+1!=-1 and j!= i :
                    area=i_areas[r]
                    b_ij = bandwidth_matrix[i + 1, j + 1] *8*pow(2,20)
                    T_send[i] += arfa * area / b_ij
                    T_receive[j] += arfa * area / b_ij
    T_trans=[x + y for x, y in zip(T_send, T_receive)]
    return T_trans

def get_max_T_t(T_inference,T_transmission):
    T = [a + b for a, b in zip(T_inference, T_transmission)]
    return max(T)

def get_T_t(T_inference,T_transmission):
    T = [a + b for a, b in zip(T_inference, T_transmission)]
    return sum(T)/len(T)

def workload_banlance_K(N,fps,X_matrix_list,Region_areas):#每个设备上的数据负载，有能耗上的考虑
    K=[0 for _ in range(N)]#每个设备要检测的面积
    for i in range(N):
        X_matrix = X_matrix_list[i]
        i_areas = Region_areas[i]
        for f in range(1, fps):
            for r in range(X_matrix.shape[1]):
                j = X_matrix[f, r] -1
                if j+1!=-1:
                    K[j]+=i_areas[r]

    K_max=pow(10,11)

    return K,np.var(K,ddof=0)/K_max#ddof=0表示总体方差#

def get_energy_consumption(T_inference,energy_list):#每个设备上的能耗 J
    K=[0 for _ in range(N)]#每个设备要检测的面积
    for i in range(N):
        K[i]+=energy_list[i+1]*T_inference[i]

    return K

def goal(X_matrix_list):#目标函数
    global T_t#ms
    T_inference=N_inference_Time(N,fps,X_matrix_list,Region_areas)
    T_transmission=N_transmission_Time(N,m0,arfa,fps,X_matrix_list,Region_areas,bandwidth_matrix)
    T_t = get_max_T_t(T_inference,T_transmission)
    if max(T_inference)>T_max or max(T_transmission)>T_max :
        return float('inf')
    #每时隙精度
    global A_t#声明修改的是全局变量
    A_t= N_average_Accuracy(N,accuracy_list,fps,X_matrix_list,Q_matrix_list)
    #每时隙系统负载均衡指数
    global K_t
    workload,K_t=workload_banlance_K(N,fps,X_matrix_list,Region_areas)

    return Query_Time_Front*(T_t - T_max)-V*(A_t-w*K_t)

def get_need_offload(X_matrix_list):
    need_offload=[]
    detect_area = [0 for _ in range(N)]  # 每个设备要检测的面积
    for i in range(N):
        X_matrix = X_matrix_list[i]
        i_areas = Region_areas[i]
        for f in range(1, fps):
            for r in range(X_matrix.shape[1]):
                j = X_matrix[f, r] - 1
                if j + 1 != -1:
                    detect_area[j] += i_areas[r]
    average_area = sum(detect_area) / N
    for i in range(N):
        if detect_area[i] > average_area:
            need_offload.append(i+1)
    return need_offload

def setrandom1(X_matrix_list_old):
    X_matrix_list = copy.deepcopy(X_matrix_list_old)

    random_i = np.random.randint(0, len(X_matrix_list))
    while np.all(X_matrix_list[random_i]== -1):
        random_i = np.random.randint(0, len(X_matrix_list))
    X_matrix = X_matrix_list[random_i]

    random_f = np.random.randint(1, X_matrix.shape[0])  # 每时隙的第一帧不参与卸载
    random_r = np.random.randint(0, X_matrix.shape[1])
    while X_matrix[random_f, random_r] == -1:
        random_f = np.random.randint(1, X_matrix.shape[0])
        random_r = np.random.randint(0, X_matrix.shape[1])

    j = X_matrix[random_f, random_r]
    random_j = np.random.randint(1, len(X_matrix_list) + 1)  # 需要保证有效更新
    while random_j == j:
        random_j = np.random.randint(1, len(X_matrix_list) + 1)  # 需要保证有效更新
    X_matrix[random_f, random_r] = random_j
    return X_matrix_list  # 返回副本

def setrandom2(X_matrix_list,need_offload):
    X_matrix_list_copy = X_matrix_list.copy()  # 创建 X_matrix 的副本

    random_i = random.choice(need_offload)
    while np.all(X_matrix_list_copy[random_i-1]== -1) :
        random_i = random.choice(need_offload)

    X_matrix = X_matrix_list_copy[random_i-1]

    random_f = np.random.randint(1, X_matrix.shape[0])  # 每时隙的第一帧不参与卸载
    random_r = np.random.randint(0, X_matrix.shape[1])
    while X_matrix[random_f, random_r] == -1:
        random_f = np.random.randint(1, X_matrix.shape[0])
        random_r = np.random.randint(0, X_matrix.shape[1])

    random_j = np.random.randint(1, len(X_matrix_list_copy) + 1)  # 需要保证有效更新
    while random_j==X_matrix[random_f, random_r]:#X_matrix[random_f, random_r]
        random_j = np.random.randint(1, len(X_matrix_list_copy) + 1)  # 需要保证有效更新
    X_matrix[random_f, random_r] = random_j
    return X_matrix_list_copy  # 返回副本

def Markov_optimization(X_matrix_list_old,tau, iterations=100):#tau=0.05,0.1
    X_matrix_list = copy.deepcopy(X_matrix_list_old) # 保存原始列表
    #need_offload = get_need_offload(X_matrix_list)
    no_improvement_count = 0
    # 计算目标函数值
    g = goal(X_matrix_list)
    g_list = [g]

    while iterations > 0:
        #new_X_matrix_list= setrandom2(X_matrix_list,need_offload)  # 随机调整列表中一个矩阵
        new_X_matrix_list = setrandom1(X_matrix_list)
        g_hat = goal(new_X_matrix_list)  # 新列表的目标函数值
        # 计算接受新解的概率 ,当(g_hat - g) / tau为正数时，eta无限趋向于0或者为0，当(g_hat - g) / tau为0时，eta为0.5
        eta = 1 / (1 + np.exp((g_hat - g) / tau))
        # 检查改进情况
        if abs(g_hat - g) < 0.01:
            no_improvement_count += 1
        else:
            no_improvement_count = 0  # 重置计数器
        # 按一定概率接受新的解
        if g_hat<g and np.random.rand() < eta:
            X_matrix_list = new_X_matrix_list  # 更新列表
            g = g_hat
            #need_offload = get_need_offload(X_matrix_list)

        g_list.append(g)
        # 停止条件
        if no_improvement_count >= 10:
            print("停止条件：目标函数值无显著改进")
            break

        iterations -= 1

    return X_matrix_list,g_list  # 返回原始列表和最终列表

if __name__ == '__main__':
    """parser = argparse.ArgumentParser(description="控制参数")
    # 添加参数解析
    #parser.add_argument("T_max", type=float, help="T_max的值")
    parser.add_argument("tau", type=float, help="tau的值")
    #parser.add_argument("w", type=float, help="w的值")

    args = parser.parse_args()
    tau=args.tau"""
    tau=0.1
    """if tau == 0.02:
        file2 = "../experiment/tau1"
    elif tau == 0.05:
        file2 = "../experiment/tau2"
    elif tau == 0.1:
        file2 = "../experiment/tau3"
    elif tau == 0.2:
        file2 = "../experiment/tau4"
    elif tau == 0.5:
        file2 = "../experiment/tau5"
    else:
        file2 = "../experiment/tau"""
    V=1
    T_max=0.65
    Qmin = 0.1
    Qmax = 3
    b_ratio=1
    w=1

    fps = 25  # 视频帧率
    l_t = 1  # 每slot长度
    alltime=180
    m0 = 1280 * 736  # 全帧的检测面积
    Color_depth = 24  # 颜色深度，用于计算每帧数据量大小 bit
    arfa = Color_depth*0.2#0.2为压缩比
    N = 4# 0代表边缘服务器
    # 精度，边缘服务器使用yolov8l，本地设备使用 #yolov8m,yolov8s,yolov8n,yolov5nu
    accuracy_list = [[0.9130,0.8702, 0.8132, 0.6976, 0.6924], [0.7394,0.7178, 0.6278, 0.4604, 0.4081],
                     [0.9097,0.8798, 0.7818, 0.7211, 0.7224], [0.7806,0.7689, 0.7229, 0.5862,0.4897]]
    phi = math.exp(-0.1)  # 衰减因子,可以随着综合指标变化
    # 每个设备的功耗，边缘服务器使用rtx3090,350w，本地使用GeForce MX230 ，15w
    energy_list=[350,15,15,15,15]
    filelist=["../traffic/vdo1/vdo1flag1.txt","../traffic/vdo2/vdo2flag1.txt","../traffic/vdo3/vdo3flag1.txt","../traffic/vdo4/vdo4flag1.txt"]
    vdo1region_area = [78848, 129024, 235520, 73728, 129024]
    vdo2region_area = [8192, 20480, 56320, 40960, 147456, 155648]
    vdo3region_area = [10240, 61440, 61440, 175104, 175104, 129024, 129024]
    vdo4region_area = [36864, 147456, 286720, 368640]
    Region_areas = [vdo1region_area,vdo2region_area,vdo3region_area,vdo4region_area]
    Region_list_all=get_Region(filelist,Region_areas,alltime,fps)
    bandwidthfile="../traffic/bandwidth/bandwidth2.log"
    bandwidth_matrix_all=get_bandwidth(bandwidthfile,N,alltime,b_ratio)

    start_time = time.time()
    Query_Time_Back=0
    Query_Time_Front=0
    A_t=0
    K_t=0
    T_t=0
    #T_max = 1 # 每片段时延阈值 0.2,0.5,1,1.5
    workloads=[0 for _ in range(N)]
    energys=[0 for _ in range(N)]
    original_K=[]
    new_K=[]
    Q_list=[]
    q_list=[]
    g_list_need=[]
    a_list=[]
    T_t_list=[]
    last_sum_T_list=[0 for _ in range(N)]

    with open("../experiment/Q_list", "w") as f:
        pass
    with open("../experiment/scheduler11.txt", "w") as file:
    #with open("../experiment/scheduler2.txt", "a") as file:
        for t in range(alltime):
            if t == 0:
                Query_Time_Front = 0  # q(0)=0
            else:
                Query_Time_Front = Query_Time_Back  # q(t)

            q_list.append(Query_Time_Front)
            Region_list = Region_list_all[t]  # 每时隙的信息
            Q=get_Q(Query_Time_Front,Qmin,Qmax)
            Q_list.append(Q)
            #Q=0.1
            with open("../experiment/Q_list", "a") as f:
                #f.write(' '.join(map(str, Q_list)) + '\n')
                f.write(str(Q) + '\n')
            X_matrix_list,Q_matrix_list = initialize_X_matrix_list1(N, fps, Region_list,Q)
            bandwidth_matrix = bandwidth_matrix_all[t]  # 每时隙的带宽信息(N+1)*(N+1)
            workload_old,K_old=workload_banlance_K(N,fps,X_matrix_list,Region_areas)
            original_K.append(K_old)

            best_X_matrix_list, g_list = Markov_optimization(X_matrix_list,tau)

            """with open(file2, "a") as f2:
                f2.write(str(g_list) + '\n')"""

            # 使用结果部署
            T_inference = N_inference_Time(N, fps, best_X_matrix_list, Region_areas)
            T_transmission = N_transmission_Time(N, m0, arfa, fps, best_X_matrix_list, Region_areas,
                                                     bandwidth_matrix)
            last_sum_T_list = [max(0,c+a+b-T_max) for a, b,c in zip(T_inference, T_transmission,last_sum_T_list)]
            T_t= get_max_T_t(T_inference,T_transmission)
            A_t = N_average_Accuracy(N, accuracy_list, fps, best_X_matrix_list, Q_matrix_list)
            workload, K_t = workload_banlance_K(N,fps,best_X_matrix_list,Region_areas)
            new_K.append(K_t)
            energy=get_energy_consumption(T_inference,energy_list)
            a_list.append(A_t)
            T_t_list.append(T_t)
            for i, l in enumerate(workload):
                workloads[i] += l
            for i, e in enumerate(energy):
                energys[i] += e
            Query_Time_Back = max(Query_Time_Front + T_t - T_max, 0)  # q (t + 1) = [q (t) + (Tt − Tmax)]+
        #file.write(f"V:{V}\n")
        #file.write(f"w:{w}\n")
        #file.write(f"q(t):{q_list}\n")
        file.write(f"original_K:{original_K}\n")
        file.write(f"new_K:{new_K}\n")
        file.write(f"q_list:{q_list}\n")
        file.write(f"Q_list:{Q_list}\n")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    slots = np.arange(0, alltime)

    # 绘制折线图
    #plt.figure(figsize=(12, 6))
    plt.plot(slots, original_K, label='original_K', color='b')
    plt.plot(slots, new_K, label='new_K', color='y')
    #plt.title('Load balancing changes',fontsize=15)
    plt.xlabel('Slots',fontsize=15)
    plt.ylabel('Load balancing index K',fontsize=15)
    plt.xticks(np.arange(0, 180, 10))
    plt.legend(fontsize=15)
    plt.savefig('D:\Desktop\Figures_pdf\ twoK.pdf', format='pdf')
    plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(slots, q_list,  color='b',label="q(t)")
    ax1.set_xlabel("Slots",fontsize=15)
    ax1.set_ylabel("q(t)", color='b',fontsize=15)  # 设置左侧纵轴标签
    ax1.tick_params(axis='y', labelcolor='b')  # 设置左侧纵轴刻度颜色
    # 创建第二个纵轴，共享横轴
    ax2 = ax1.twinx()
    ax2.plot(slots, Q_list, color='m', label="Q")
    ax2.set_ylabel("Q", color='m',fontsize=15)  # 设置右侧纵轴标签
    ax2.tick_params(axis='y', labelcolor='m')  # 设置右侧纵轴刻度颜色
    # Add legends for both axes
    ax1.legend(loc='upper left', fontsize=15)  # Legend for ax1
    ax2.legend(loc='upper right', fontsize=15)  # Legend for ax2
    # 设置标题
    #plt.title("q(t) and Q threshold per slot", fontsize=15)
    plt.savefig('D:\Desktop\Figures_pdf\qQ.pdf', format='pdf')
    plt.show()
