import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from DNABERT2_MIX.bert_layers import BertModel

class DNABert_S(nn.Module):
    def __init__(self, feat_dim=128, load_dict=None):
        super(DNABert_S, self).__init__()
        print("-----Initializing DNABert_S-----")
#        self.dnabert2 = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.dnabert2 = BertModel.from_pretrained("zhihan1996/DNABERT-2-117M")
        self.emb_size = self.dnabert2.pooler.dense.out_features
        self.feat_dim = feat_dim

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))
        if load_dict != None:
            self.dnabert2.load_state_dict(torch.load(load_dict+'pytorch_model.bin'))  
            self.contrast_head.load_state_dict(torch.load(load_dict+'head_weights.ckpt'))
        
    def forward(self, input_ids, attention_mask, task_type='train'):        
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
        else:
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1) 
            bert_output_1 = self.dnabert2.forward(input_ids=input_ids_1, attention_mask=attention_mask_1)
            bert_output_2 = self.dnabert2.forward(input_ids=input_ids_2, attention_mask=attention_mask_2)
            attention_mask_1 = attention_mask_1.unsqueeze(-1)
            attention_mask_2 = attention_mask_2.unsqueeze(-1)
            mean_output_1 = torch.sum(bert_output_1[0]*attention_mask_1, dim=1) / torch.sum(attention_mask_1, dim=1)
            mean_output_2 = torch.sum(bert_output_2[0]*attention_mask_2, dim=1) / torch.sum(attention_mask_2, dim=1)
            
            cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
            
            return cnst_feat1, cnst_feat2, mean_output_1, mean_output_2
            
    def contrast_logits(self, embd1, embd2):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        feat2 = F.normalize(self.contrast_head(embd2), dim=1)
        return feat1, feat2

    # calculate the embedding of an input sequence as the average embeddings of its tokens
    def get_mean_embeddings(self, input_ids, attention_mask):
        # mean embeddings
        bert_output = self.dnabert2(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return embeddings