import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, OPTModel, GPT2Tokenizer
from datasets import load_dataset, load_from_disk
class opt_retreiver(torch.nn.Module):
    def __init__(self,weight_path):
        super(opt_retreiver,self).__init__()
        self.dim = 2048
        self.device = torch.device("cuda:0")
        self.weights_path = weight_path
        
        self.lm_model = OPTModel.from_pretrained(self.weights_path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.weights_path)
        self.header = nn.Linear(self.dim,self.dim).to(self.device)
        
        self.lm_out = None 
        self.model_out = None 
        self.loss = None 
        self.lr = 1e-6
        self.optimizer = optim.Adam(self.parameters(),lr= self.lr) 
    # def _forward(self,inputs):
    #     input_ids = inputs.to(self.device)
    #     self.lm_out = self.lm_model(input_ids = input_ids)['last_hidden_state'][0][-1]
    #     self.model_out = self.header(self.lm_out) 
    #     return self.model_out

    def main_train(self,dataloader:DataLoader):
        self.lr = 1e-6
        self.epoches = 1
        self.optimizer = optim.Adam(self.parameters(),lr= self.lr)
        for n in range(self.epoches):
            for i,x in enumerate(dataloader):
                loss = self.train_batch_forward(x)
                self.train_batch_backward(loss)
                print(loss)
    
    def forward(self,inputs):
        # self.lm_out = self.lm_model(input_ids = inputs['input_ids'].to(self.device), attention_mask = inputs['attention_mask'].to(self.device))['last_hidden_state'][:,-1,:].detach()
        # self.model_out = self.header(self.lm_out)
        # return self.model_out
        lm_out = self.lm_model(input_ids = inputs['input_ids'].to(self.device), attention_mask = inputs['attention_mask'].to(self.device))['last_hidden_state'][:,-1,:].detach()
        model_out = self.header(lm_out)
        return model_out
            
    def dot_product_scores(self,q_vectors: torch.tensor, ctx_vectors: torch.tensor) -> torch.tensor:
        """
        calculates q->ctx scores for every row in ctx_vector
        :param q_vector:
        :param ctx_vector:
        :return:
        """
        # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
        r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
        return r

    def NLLloss(self,q_vectors:torch.tensor, ctx_vectors: torch.tensor, pos_ids: torch.tensor):
        scores = self.dot_product_scores(q_vectors, ctx_vectors)
        softmax_scores = torch.nn.functional.log_softmax(scores, dim=1)
        loss = torch.nn.functional.nll_loss(
                softmax_scores,
                pos_ids,
                reduction="mean",
        )
        return loss
    
    def train_batch_forward(self, batch_data):
        size = len(batch_data[0])
        temp_loss = 0
        for i in range(size):
            temp_loss += self.train_data_forward(batch_data[0][i], batch_data[1][i])
        return temp_loss/size


    def train_data_forward(self,input_text:str, ctx_vectors: torch.tensor):
        inputs = self.tokenizer(input_text,return_tensors = 'pt')
        return self.train_forward(inputs, ctx_vectors)
    
    def train_forward(self,inputs, ctx_vectors:torch.tensor):
        q_vectors = self.forward(inputs).to(self.device)
        ctx_vectors = ctx_vectors.to(self.device)
        pos_ids = torch.tensor([0]*int(q_vectors.size()[0])).to(self.device)
        loss = self.NLLloss(q_vectors,ctx_vectors,pos_ids)
        return loss 
    
    def train_batch_backward(self,loss:torch.tensor):
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.zero_grad()


