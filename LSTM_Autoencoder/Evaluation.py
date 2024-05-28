import csv
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt


#作圖
def plot_vibration(timeaxis, data, n=1, start_point=0, end_poins=150):
    plt.figure(n)
    plt.plot(timeaxis[:end_poins-start_point], data[start_point:end_poins])
    plt.xlabel('t(s)')
    plt.ylabel('phase(\u03C6)')
    plt.title('unwrapphase')
    plt.show()


######load data
###load data_pre.npz
plt.close('all')
dataset = np.load('../data/data_pre.npz') 
pre_normal = dataset['normal']
pre_anomaly = dataset['anomaly']
pre_normal= np.squeeze(pre_normal,axis=1)  #降維
pre_anomaly= np.squeeze(pre_anomaly,axis=1) #降維
#print(pre_normal.shape)
#print(pre_anomaly.shape)

total = (len(pre_anomaly) + len(pre_normal))
print(total)


#########confusion_matrix的參數
threshold = 23
TP_count = 0
FN_count = 0
TN_count = 0
FP_count = 0

y_true_label = [] #0~209 normal #210~249 anomaly
y_pred_label = []
y_score =[]

for row in pre_normal :
    #print(row)
    y_true_label.append(0)
    y_score.append(float(row))
    if float(row) <= threshold :
        TP_count = TP_count+1
        y_pred_label.append(0)
    else :
        FN_count = FN_count +1
        y_pred_label.append(1)
        
print(TP_count,FN_count,TN_count,FP_count)
for row in pre_anomaly :
    #print(row[1])
    y_true_label.append(1)
    y_score.append(float(row))
    if float(row) > threshold :          
        TN_count = TN_count +1
        y_pred_label.append(1)
    else :
        FP_count = FP_count +1
        y_pred_label.append(0)

print(TP_count,FN_count,TN_count,FP_count)
y_true_label = np.array(y_true_label)
y_pred_label = np.array(y_pred_label)
y_score = np.array(y_score)

#預測值
TPR = TP_count / (TP_count + FN_count)
FPR = FP_count / (TN_count + FP_count)
recall = TPR

precision = TP_count / (TP_count + FP_count)
F1_score = (2*precision*recall)/(precision + recall)
acc = (TP_count+TN_count)/ total

print('TPR =',TPR)
print('FPR =',FPR)
print('acc =',acc)
print('precision =',precision)
print('F1_score =',F1_score)

########################confusion_matrix的圖

cf_matrix = confusion_matrix(y_true_label, y_pred_label)  
disp = ConfusionMatrixDisplay(confusion_matrix=cf_matrix,display_labels=['normal','anomaly'])

disp = disp.plot()
disp.ax_.set_title("Confusion matrix")
#plt.savefig('Confusion_matrix.png')

###confusion_matrix的圖中顯示TPR、FPR

#cmn = cf_matrix.diagonal()/cf_matrix.sum(axis=0) 
cmn = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[ :,np.newaxis]
disp = ConfusionMatrixDisplay(confusion_matrix=cmn,display_labels=['normal','anomaly'])

# NOTE: Fill all variables here with default values of the plot_confusion_matrix
disp = disp.plot()
disp.ax_.set_title("confusion matrix rate")
#plt.savefig('Confusion_matrix_rate.png')


###########################AUC 、ROC_Curve
plt.figure()
fpr, tpr, thresholds = roc_curve(y_true_label ,y_score,pos_label = 0) #pos_label  意思是標籤0的是正常樣本 #drop_intermediate=False (全部顯示)

#print(fpr)
#print(tpr)
fpr_radar = 1-fpr
tpr_radar = 1-tpr
#print(fpr_radar)  
#print(tpr_radar)
#print(thresholds)


roc_auc = auc(fpr_radar,tpr_radar)
print('roc_auc =',roc_auc)

plt.plot(fpr_radar,tpr_radar,'k--',label ='ROC(area = {0:.2f})'.format(roc_auc), lw=2)
plt.title('ROC Curve')
plt.xlabel('False Positive Rate(FPR)')
plt.ylabel('True Positive Rate(TPR)')
plt.legend(loc='lower right')
#plt.savefig('ROC_Curve.png')
plt.show()