import json 
import torch 
from datasets import load_dataset,load_from_disk
import random
from transformers import GPT2Tokenizer,OPTModel 
from torch.utils.data import Dataset 


class retreiver_dataset(Dataset):
    def __init__(self,path):
        self.hf_dataset = load_from_disk(path)
    def __len__(self):
        return len(self.hf_dataset)
    def __getitem__(self,idx):
        q_text = self.hf_dataset[idx]['question']
        c_emds = torch.tensor(self.hf_dataset[idx]['context'])
        return (q_text, c_emds)



def create_neg_ids(pos_id:int, num_data:int,num_gen:int):
    i = 0
    result = []
    while(i<num_gen):
        temp = random.randint(0,num_data-1)
        if(temp in result):
            continue 
        else:
            result.append(temp)
            i+=1 
    return result

def convert_squad_dic():
    device = torch.device("cuda:1")
    tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-350m')
    opt_model = OPTModel.from_pretrained('facebook/opt-1.3b').to(device)
    squad_dataset = load_dataset('squad')
    # question_list = []
    # context_list = []
    total_list = [] 
    size_squad = len(squad_dataset['train'])
    for i in range(size_squad):
        psg_text = []
        psg_emd = []
        # question_list.append(squad_dataset['train'][i]['question'])
        psg_text.append(squad_dataset['train'][i]['context'])
        neg_psg_ids = create_neg_ids(pos_id=i, num_data= size_squad, num_gen= 10)
        for j in neg_psg_ids:
            psg_text.append(squad_dataset['train'][j]['context'])
        for text in psg_text:
            input_ids = tokenizer(text,return_tensors="pt")['input_ids'].to(device)
            emd = opt_model(input_ids = input_ids)['last_hidden_state'].cpu().detach()[0][-1]
            psg_emd.append(emd)
        psg_emd = torch.stack(psg_emd)
        total_list.append({'question':squad_dataset['train'][i]['question'],'context': psg_emd.tolist()})
        print("Current: ", i, "total: ", size_squad)
        # context_list.append(psg_emd.tolist())
    return {"data": total_list}
    # return {"question": question_list, "context": context_list}

def convert_dic_json(dic:dict,path):
    with open(path, "w") as outfile:
        json.dump(dic, outfile)

def convert_json_dataset(json_path,save_path):
    temp_dataset = load_dataset("json", data_files=json_path, split="train",field = "data")
    temp_dataset.save_to_disk(save_path)

def main_squad_json(path = "../data/json_data/sample.json"):
    dic = convert_squad_dic()
    convert_dic_json(dic,path)

def main_squad_json_dataset(json_path="../data/json_data/retreiver.json", dataset_path = "../data/convert_dataset/retreiver" ):
    dic = convert_squad_dic()
    convert_dic_json(dic,json_path)
    convert_json_dataset(json_path,dataset_path)

    

def load_json_dataset(path):
    temp_dataset = load_dataset("json", data_files=path, split="train",field = "data")
    return temp_dataset


if __name__ == "__main__":
    main_squad_json_dataset()
    # main_squad_json(path = "../data/json_data/sample.json")


