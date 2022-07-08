import torch 
import torch.nn as nn 
from transformers import AutoModel, OPTModel, GPT2Tokenizer
from datasets import load_dataset, load_from_disk
class opt_retreiver(torch.nn.Module):
    def __init__(self):
        super(opt_retreiver,self).__init__()
        self.dim = 2048
        self.device = torch.device("cuda:0")
        self.weights_path ='/raid/projects/wentaoy4/model_file/models--facebook--opt-1.3b/snapshots/c8fd4232a5df1e87e06d5cbb9e066c5a114cd4ee'
        
        self.lm_model = OPTModel.from_pretrained(self.weights_path).to(self.device)
        self.header = nn.Linear(self.dim,self.dim).to(self.device)
        
        self.lm_out = None 
        self.model_out = None 
        
        self.loss = None 
        
    # def _forward(self,inputs):
    #     input_ids = inputs.to(self.device)
    #     self.lm_out = self.lm_model(input_ids = input_ids)['last_hidden_state'][0][-1]
    #     self.model_out = self.header(self.lm_out) 
    #     return self.model_out
    
    def _forward(self,inputs):
        self.lm_out = self.lm_model(input_ids = inputs['input_ids'].to(self.device), attention_mask = inputs['attention_mask'].to(self.device))['last_hidden_state'][:,-1,:]
        self.model_out = self.header(self.lm_out)
        return self.model_out
            
        
    
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
    
    def train_forward(self,inputs, ctx_vectors:torch.tensor):
        q_vectors = self._forward(inputs).to(self.device)
        ctx_vectors = ctx_vectors.to(self.device)
        pos_ids = torch.tensor([0]*int(q_vectors.size()[0])).to(self.device)
        self.loss = self.NLLloss(q_vectors,ctx_vectors,pos_ids)
        return self.loss 
    
    def train_backward(self,loss):
        
    
    
    def batch_train(self):