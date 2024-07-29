import datetime
import random
import torch
import Training
import numpy as np
import Model
import pandas as pd
import warnings
import psutil
import time
def wait_process(p):
    pid_list = psutil.pids()
    wait_pid = p  # 等待的进程号
    while wait_pid in pid_list:
        # print('still working!' + str(wait_pid))
        time.sleep(2)
        pid_list = psutil.pids()
warnings.filterwarnings('ignore')




def load_dataset(sentence_path,label_path,YY_label_path,matrix_path):
    # content,label=[],[]
    sen=pd.read_pickle(sentence_path)
    lab=pd.read_pickle(label_path)
    YYlab=pd.read_pickle(YY_label_path)
    matrixs=pd.read_pickle(matrix_path)
    return sen,lab,YYlab,matrixs

def load_dataset_valid(sentence_path,label_path,YY_label_path,matrix_path):
    sen = pd.read_pickle(sentence_path)
    lab = pd.read_pickle(label_path)
    YYlab = pd.read_pickle(YY_label_path)
    matrixs = pd.read_pickle(matrix_path)
    valid_sen=sen[::10]
    valid_label=lab[::10]
    valid_YYlabel=YYlab[::10]
    valid_matrixs=matrixs[::10]
    return valid_sen,valid_label,valid_YYlabel,valid_matrixs


# wait_process(3611782)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=Model.Xlnet_Encoder(device).to(device)

seed_num=2
np.random.seed(seed_num)
torch.manual_seed(seed_num)
torch.cuda.manual_seed(seed_num)
torch.cuda.manual_seed_all(seed_num)
random.seed(seed_num)
os.environ['PYTHONHASHSEED'] = str(seed_num)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled=False


# train_data=load_dataset('data_add_data/train_sentence.pickle', 'data_add_data/train_labels.pickle',
#                         'data_add_data/train_yylabels.pickle')
train_data=load_dataset('../dadd/train_sentence.pickle','../dadd/train_labs.pickle','../dadd/train_yylabs.pickle','../dadd/train_matrix.pickle')
# train_data=load_dataset('../data_run/Train_InputSentences.pickle','../data_run/Train_labels.pickle','../data_run/Train_Yuyong_labels.pickle','../data_run/train_matrixs.pickle')
test_data=load_dataset('../data/Test_InputSentences.pickle','../data/Test_labels.pickle','../data/Test_Yuyong_labels.pickle','../predict_matrix/test_matrixs.pickle')
valid_data=load_dataset_valid('../data/Train_InputSentences.pickle','../data/Train_labels.pickle','../data/Train_Yuyong_labels.pickle','../data/train_matrixs.pickle')

print('train_data:',len(train_data[0]))
print('test_data:',len(test_data[0]))
print('valid_data:',len(valid_data[0]))

start_time=datetime.datetime.now()
Training.train(model,train_data,valid_data)
train_end=datetime.datetime.now()
print('训练时间：',train_end-start_time)
Training.test(model,test_data)
end_time=datetime.datetime.now()
print('测试时间：',end_time-train_end)

print('开始时间：',start_time)
print('结束时间：',end_time)
print('总用时：',end_time-start_time)