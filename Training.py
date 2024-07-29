import datetime

import torch
import torch.nn as nn
from transformers import AdamW
from torch.optim import Adam
import numpy as np
from segeval.window.pk import pk
import torch.nn.functional as F
from sklearn import metrics
from IPython import embed
import Adversarial
import torch.optim.lr_scheduler as lr

learning_rate=1e-5
running_epoch=30
target_class=['Background', 'Behavior', 'Cause', 'Comment', 'Contrast', 'Illustration', 'Lead', 'Progression', 'Purpose', 'Result', 'Situation', 'Statement', 'Sub-Summary', 'Sumup', 'Supplement']


class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # 此处的self.smoothing即我们的epsilon平滑参数。

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


loss_function=LabelSmoothing(0.1)


def train(model,train_data,valid_data):
    optimizer=Adam(model.parameters(), lr=learning_rate, eps=1e-8)
    
    scheduler_1 = lr.MultiStepLR(optimizer, milestones=[2000, 2500], gamma=0.5)
    
    dev_best_loss = 1000
    for epoch in range(running_epoch):
        start_time=datetime.datetime.now()
        num_iter = 0
        print('Epoch[{}/{}]'.format(epoch + 1, running_epoch))
        state = np.random.get_state()
        np.random.shuffle(train_data[0])
        np.random.set_state(state)
        np.random.shuffle(train_data[1])
        np.random.set_state(state)
        np.random.shuffle(train_data[2])
        np.random.set_state(state)
        np.random.shuffle(train_data[3])
        
        
        # fgm=Adversarial.FGM(model)
        pgd=Adversarial.PGD(model)
        k=3
        for sentence,seglabel,yylabel,matrix in zip(train_data[0],train_data[1],train_data[2],train_data[3]):
            seglabel=torch.tensor(seglabel).cuda()
            yylabel = torch.tensor(yylabel).cuda()
            matrix=torch.tensor(matrix,dtype=int).cuda()
            model.train()
            seg_out, class_out=model(sentence,seglabel,matrix,'Training')
            yy_loss=F.cross_entropy(class_out,yylabel)
            seg_loss=F.cross_entropy(seg_out,seglabel)
            total_loss=seg_loss+yy_loss
            optimizer.zero_grad()
            total_loss.backward()

            # fgm.attack()
            # seg_out, class_out = model(sentence,seglabel,matrix,'Training')
            # yy_loss = F.cross_entropy(class_out, yylabel)
            # seg_loss = F.cross_entropy(seg_out, seglabel)
            # total_loss = seg_loss + yy_loss
            # total_loss.backward()
            # fgm.restore()
            
            pgd.backup_grad()
            for t in range(k):
                pgd.attack(is_first_attack=(t==0))
                if t!=k-1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                seg_out, class_out=model(sentence,seglabel,matrix,'Training')
                yy_loss=F.cross_entropy(class_out,yylabel)
                seg_loss=F.cross_entropy(seg_out,seglabel)
                total_loss=seg_loss+yy_loss
                total_loss.backward()
            pgd.restore()
            
            
            optimizer.step()
            scheduler_1.step()
            model.zero_grad()
            if num_iter % 20 == 0:
                true = yylabel.data.cpu()
                predict = torch.max(class_out.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)
                # dev_acc, dev_loss, dev_pk = evaluate(model, valid_data)
                # if dev_loss < dev_best_loss:
                #     dev_best_loss = dev_loss
                #     torch.save(model.state_dict(), 'saved_model/best.ckpt')
                # print(
                #     'Iter:{iter},train_loss:{t_l:.2} , train_acc:{t_a:.2} , length:{len} , valid_loss:{v_l:.2} , valid_acc:{v_a:.2},valid_pk:{v_p:.2}'.format(
                #         iter=num_iter, t_l=total_loss.item(), t_a=train_acc, len=len(seglabel), v_l=dev_loss, v_a=dev_acc,
                #         v_p=dev_pk))
                print(
                    'Iter:{iter},train_loss:{t_l:.2} , train_acc:{t_a:.2} , length:{len} '.format(
                        iter=num_iter, t_l=total_loss.item(), t_a=train_acc, len=len(seglabel)))
            num_iter += 1
        end_time=datetime.datetime.now()
        print('one_epoch cost time:',end_time-start_time)

        torch.save(model.state_dict(),'saved_model/epoch{my_epoch}.ckpt'.format(my_epoch=epoch))


def evaluate(model,valid_data):
    model.eval()
    loss_total=0
    pk_total=0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for sentence,seglabel,yylabel,matrix in zip(valid_data[0],valid_data[1],valid_data[2],valid_data[3]):
            seglabel = torch.tensor(seglabel).cuda()
            yylabel = torch.tensor(yylabel).cuda()
            matrix=torch.tensor(matrix,dtype=int).cuda()
            seg_out, class_out= model(sentence,seglabel,matrix)
            predict_seglabel=torch.max(seg_out.data,1)[1].cpu()
            seg_loss=F.cross_entropy(seg_out,seglabel)
            yy_loss = F.cross_entropy(class_out, yylabel)
            total_loss = seg_loss + yy_loss
            loss_total+=total_loss
            yylabel = yylabel.data.cpu().numpy()
            yypredict = torch.max(class_out.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, yylabel)
            predict_all = np.append(predict_all, yypredict)
            gold=get_pk_index(seglabel)
            predict=get_pk_index(predict_seglabel)
            pk_total+=pk(predict,gold,one_minus=True)

    acc = metrics.accuracy_score(labels_all, predict_all)
    ave_loss=loss_total/len(valid_data[0])
    ave_pk=pk_total/len(valid_data[0])
    return acc,ave_loss,ave_pk

def test(model,test_data):
    for i in range(running_epoch):
        model.load_state_dict(torch.load('saved_model/epoch{num}.ckpt'.format(num=i)))
        print('epoch_{}'.format(i))
        model.eval()
        loss_total=0
        pk_total=0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for sentence, seglabel, yylabel,matirx in zip(test_data[0], test_data[1], test_data[2], test_data[3]):
                seglabel = torch.tensor(seglabel).cuda()
                yylabel = torch.tensor(yylabel).cuda()
                matirx = torch.tensor(matirx, dtype=int).cuda()
                seg_out, class_out= model(sentence,seglabel,matirx)
                predict_seglabel=torch.max(seg_out.data,1)[1].cpu()
                yy_loss = F.cross_entropy(class_out, yylabel)
                seg_loss=F.cross_entropy(seg_out,seglabel)
                total_loss = seg_loss + yy_loss
                loss_total += total_loss
                yylabel = yylabel.data.cpu().numpy()
                yypredict = torch.max(class_out.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, yylabel)
                predict_all = np.append(predict_all, yypredict)
                gold = get_pk_index(seglabel)
                predict = get_pk_index(predict_seglabel)
                pk_total += pk(predict, gold, one_minus=True)
        acc=metrics.accuracy_score(labels_all, predict_all)
        report = metrics.classification_report(labels_all, predict_all, target_names=target_class, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        ave_loss = loss_total / len(test_data[0])
        ave_pk = pk_total / len(test_data[0])
        msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}, 1-PK:{2:>5.4}'
        print(msg.format(ave_loss, acc, ave_pk))
        print('Precision, Recall and F1-Score')
        print(report)
        print('Confusion Maxtrix')
        print(confusion)
        print('*' * 20)
    # model.load_state_dict(torch.load('saved_model/best.ckpt'))
    # model.eval()
    # loss_total = 0
    # pk_total = 0
    # predict_all = np.array([], dtype=int)
    # labels_all = np.array([], dtype=int)
    # with torch.no_grad():
    #     for sentence, seglabel, yylabel,matrix in zip(test_data[0], test_data[1], test_data[2],test_data[3]):
    #         seglabel = torch.tensor(seglabel).cuda()
    #         yylabel = torch.tensor(yylabel).cuda()
    #         matirx = torch.tensor(matirx,dtype=int).cuda()
    #         seg_out, class_out = model(sentence,seglabel,matirx)
    #         predict_seglabel = torch.max(seg_out.data, 1)[1].cpu()
    #         yy_loss = F.cross_entropy(class_out, yylabel)
    #         seg_loss = F.cross_entropy(seg_out, seglabel)
    #         total_loss = seg_loss + yy_loss
    #         loss_total += total_loss
    #         yylabel = yylabel.data.cpu().numpy()
    #         yypredict = torch.max(class_out.data, 1)[1].cpu().numpy()
    #         labels_all = np.append(labels_all, yylabel)
    #         predict_all = np.append(predict_all, yypredict)
    #         gold = get_pk_index(seglabel)
    #         predict = get_pk_index(predict_seglabel)
    #         pk_total += pk(predict, gold, one_minus=True)
    # acc = metrics.accuracy_score(labels_all, predict_all)
    # report = metrics.classification_report(labels_all, predict_all, target_names=target_class, digits=4)
    # confusion = metrics.confusion_matrix(labels_all, predict_all)
    # ave_loss = loss_total / len(test_data[0])
    # ave_pk = pk_total / len(test_data[0])
    # msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}, 1-PK:{2:>5.4}'
    # print(msg.format(ave_loss, acc, ave_pk))
    # print('Precision, Recall and F1-Score')
    # print(report)
    # print('Confusion Maxtrix')
    # print(confusion)
    # print('*' * 20)


def get_pk_index(labels):
    label_list=[]
    last_index=-1
    for index, i in enumerate(labels):
        if i == 1:
            label_list.append(index - last_index)
            last_index = index
    if labels[-1]==0:
        label_list.append(len(labels)-sum(label_list))
    return label_list
