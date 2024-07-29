from Encoder import Encoder,PositionalEncoding
import torch
import torch.nn as nn
from transformers import AutoModel,AutoTokenizer
import torch.nn.functional as F
from GCN1 import GraphEncoder
from typing import List
import pandas as pd
from IPython import embed
CLS,SEP='[CLS]','[SEP]'


class Xlnet_Encoder(nn.Module):
    def __init__(self,device):
        super(Xlnet_Encoder, self).__init__()
        self.embed_dim=768
        self.hidden_size=768
        self.device=device
        self.xlnet=AutoModel.from_pretrained('../xlnet_base')
        self.tokenizer=AutoTokenizer.from_pretrained('../xlnet_base')
        self.encode=Encoder(device=self.device,n_layers=12,d__model=self.embed_dim, d_k=self.embed_dim, d_v=self.embed_dim, h=8)
        # self.PositionalEncoding=PositionalEncoding(d_model=self.embed_dim)
        self.dropout=nn.Dropout(p=0)
        self.seg_classifier=nn.Sequential(
            nn.Tanh(),
            nn.Linear(self.embed_dim,2)
        )

        self.classifier=nn.Sequential(
            nn.Tanh(),
            nn.Linear(3*self.embed_dim,15)
        )

        self.ws1 = nn.Linear(self.hidden_size , self.hidden_size)
        self.ws2 = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)


        self.gcn=GraphEncoder(hdim=self.hidden_size,device=self.device,nlayers=2)


    def forward(self,sentence,gold,graph,lab='Test'):
        #     get_xlnet_represent1()
        # print(len(sentence))
        embedding,sep_index=self.get_xlnet_represent1(sentence)
        sen_vec_list = []
        for index in sep_index:
            sen_vec_list.append(embedding[:, index, :])
        sen_vec = torch.stack(sen_vec_list).permute(1,0,2)   #[batch,sen_num,embed_dim]
        # print(sen_vec.shape)

        #     get_xlnet_represent()
        # sen_vec=self.get_xlnet_represent(sentence)
        #gcn
        disco_mask = (sen_vec[:, :, 0] >= 0).long()
        gra=self.create_disco_graph(graph,len(graph))
        gra_out = self.gcn.transform_sent_rep(sen_vec, disco_mask, gra)

        encoder_input=self.dropout(sen_vec)
        out=self.encode(encoder_input)        #[batch,sen_num,embed_dim]
        seg_input=out.squeeze(0)        #[sen_num,embed_dim]
        seg_out=self.seg_classifier(seg_input)        #[sen_num,class_num]
        pre_seg_index=torch.max(seg_out.data,1)[1]
        if lab=='Training':
            seg_embedding=self.get_docseg_embedding(sen_vec,gold)   #[1,sen_num,embed_dim]       get_docseg_embedding(a,b),a[sen_num,batch,embed_dim]
        else:
            seg_embedding=self.get_docseg_embedding(sen_vec,pre_seg_index)

        combine_embedding=torch.cat((out,seg_embedding),2)  #[1,sen_num,2*embedding]
        combine_embeddings=torch.cat((combine_embedding,gra_out),2).squeeze(0) #[1,sen_num,3*embedding]->[sen_num,3*embedding]
        class_out=self.classifier(combine_embeddings)              #[sen_num,class_num]

        return seg_out,class_out



    def get_xlnet_represent1(self, content):
        output = []
        sep_index = []
        for sentence in content:
            token = self.tokenizer.tokenize(sentence)
            token = token + ['<sep>']
            output += token
            sep_index.append(len(output) - 1)
        output.append('<cls>')
        token_id = self.tokenizer.convert_tokens_to_ids(output)
        token_tensor = torch.tensor([token_id]).to(self.device)
        embedding = self.xlnet(token_tensor)[0]
        return embedding, sep_index

    def get_xlnet_represent(self,sentence):
        cls_vec=[]
        for sentence in sentence:
            token = self.tokenizer.tokenize(sentence)
            token = token + ['<sep>'] + ['<cls>']
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            token_tensor = torch.tensor([token_id]).to(self.device)
            word_vec = self.xlnet(token_tensor)[0]
            cls_vec.append(word_vec[:, -1, :])
        cls_out = torch.stack(cls_vec)
        return cls_out.permute(1,0,2)

    def get_docseg_embedding(self, sen_embedding, label):
        #训练的预测标签最后一个标签可能不为1
        if label[-1]==0:
            label[-1]=1
        #分割位置
        seg_point = []
        #每个分割块长度
        seg_num=[]
        t=-1
        for index, i in enumerate(label):
            if i == 1:
                seg_point.append(index+1)
                seg_num.append(index-t)
                t=index
        seg_vec_list = []
        if len(seg_point)>1:
            for i in range(len(seg_point)-1):
                if i == 0:
                    seg_vec_list.append(sen_embedding[:,0:seg_point[i], :])
                else:
                    seg_vec_list.append(sen_embedding[: ,seg_point[i - 1]:seg_point[i], :])
            seg_vec_list.append(sen_embedding[: ,seg_point[-2]:,  :])
        else:
            seg_vec_list.append(sen_embedding)
        seg_vec=[]
        # print(seg_vec_list)
        for tens in seg_vec_list:
            seg_vec.append(self.get_segmentation_embedding(tens))
        final_seg_out=[]
        for index,i in enumerate(seg_vec):
            for num in range(seg_num[index]):
                final_seg_out.append(i)
        final_seg_out=torch.stack(final_seg_out)
        return final_seg_out.permute(1,0,2)

    def get_segmentation_embedding(self,sen_output):
        self_attention = torch.tanh(self.ws1(sen_output))  # [sen_num,sen_len,2*hidden_dim]
        self_attention = self.ws2(self_attention).squeeze(2)  # [sen_num,sen_len]
        self_attention = self.softmax(self_attention)
        sent_encoding_ = torch.sum(sen_output * self_attention.unsqueeze(-1), dim=1)  # [sen_num,2*hidden_dim]
        return sent_encoding_

    def create_disco_graph(self,disco_graph, num_of_disco: int) -> List[tuple]:
        ########
        # disco graph
        dis_graph_as_list_of_tuple = []
        # dis_graph = np.zeros((num_of_disco, num_of_disco), dtype=float)
        for rst in disco_graph:
            rst_src, rst_tgt = rst[0], rst[1]
            if rst_src < num_of_disco and rst_tgt < num_of_disco:
                # dis_graph[rst_src][rst_tgt] = 1
                dis_graph_as_list_of_tuple.append((rst_src, rst_tgt))
        # dis_graph = ArrayField(dis_graph, padding_value=0, dtype=np.float32)

        return dis_graph_as_list_of_tuple

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# sentence=['没什么比这个更绝望，找不到好的模型','我的心态爆炸了啊','哎呦不错哦，小老弟，干得漂亮','你可真是个憨憨，做了些什么鬼东西']
# matrix=pd.read_pickle('train_matrixs.pickle')
# g=torch.tensor(matrix[0],dtype=int)
# gold=torch.tensor([0,1,0,1])
#
# model=Xlnet_Encoder(device).to(device)
# out=model(sentence,gold,g)
# # print(out.shape)
# for i in out:
#     print(i.shape)