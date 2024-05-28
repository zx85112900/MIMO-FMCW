import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def normalize_data(y_data, yn_min, yn_max):          
    y_min = np.min(y_data, axis = 1)
    y_max = np.max(y_data, axis = 1)
    #print('y_min = ',y_min)
    #print('y_max = ',y_max)
    a = (yn_max - yn_min)/(y_max - y_min) 
    b = yn_min - a * y_min   
    a = np.expand_dims(a,axis=1) #升維
    b = np.expand_dims(b,axis=1) #升維
    yn_data = a * y_data +b
    return yn_data

# loss函數(均方誤差)  
def mse_loss(target, shifted):
    return np.mean((target - shifted)**2)

#作圖
def plot_vibration(timeaxis, data, n=1, start_point=0, end_poins=150):
    plt.figure(n)
    plt.plot(timeaxis[:end_poins-start_point], data[start_point:end_poins])
    plt.xlabel('t(s)')
    plt.ylabel('phase(\u03C6)')
    plt.title('unwrapphase')
    plt.show()

###################main

#####load
dataset = np.load('../data/data_finger.npz')
dataset_normal = dataset['normal']
dataset_anomaly = dataset['anomaly']
print(dataset_normal.shape)
print(dataset_anomaly.shape)


#SG濾波器
smoothed_normal_data = savgol_filter(dataset_normal, window_length=7, polyorder=2)
smoothed_anomaly_data = savgol_filter(dataset_anomaly, window_length=7, polyorder=2)
#min-max正規化
normalize_normal_data = normalize_data(smoothed_normal_data,-1,1)
normalize_anomaly_data = normalize_data(smoothed_anomaly_data,-1,1)


############################ 找loss值，做資料偏移
# 初始化偏移量列表
line_data = normalize_normal_data[210,0:150]
shifts = [0] * len(normalize_normal_data)
shifts_anomaly = [0] * len(normalize_anomaly_data)
shift_normal = np.zeros((0,140), dtype=np.float32)
shift_anomaly = np.zeros((0,140), dtype=np.float32)

for i in range(len(normalize_normal_data)):
    best_shift = shifts[i]
    best_loss = 10000
    
    for shift in range(0,10):
        current_loss = mse_loss(line_data[0:140], normalize_normal_data[i,shift:140+shift])
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_shift = shift
    
    # 更新偏移量列表
    shifts[i] = best_shift
    temp = normalize_normal_data[i,best_shift:150]
    temp = np.expand_dims(temp[0:140],axis=0) #axis=1
    shift_normal = np.append(shift_normal,temp,axis=0) 
    
#############################anomaly
for i in range(len(normalize_anomaly_data)):
    best_shift = shifts_anomaly[i]
    best_loss = 10000
    
    for shift in range(0,10):
        current_loss = mse_loss(line_data[0:140], normalize_anomaly_data[i,shift:140+shift])
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_shift = shift
    
    # 更新偏移量列表
    shifts[i] = best_shift
    temp = normalize_anomaly_data[i,best_shift:150]
    temp = np.expand_dims(temp[0:140],axis=0) #axis=1
    shift_anomaly = np.append(shift_anomaly,temp,axis=0) 

np.savez('../data/data_process.npz',normal=shift_normal,anomaly=shift_anomaly)
print(shift_normal.shape)
print(shift_anomaly.shape)

##################作圖
#參數
delta_t = 0.05
frame_len = len(dataset_normal[0])
timeaxis=np.arange(0, frame_len*delta_t, delta_t)







