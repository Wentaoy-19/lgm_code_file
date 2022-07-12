import json 
import torch 
from datasets import load_dataset 
import random
from transformers import GPT2Tokenizer,OPTModel 

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
    tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-350m')
    opt_model = OPTModel.from_pretrained('facebook/opt-1.3b')
    squad_dataset = load_dataset('squad')
    question_list = []
    context_list = []
    size_squad = len(squad_dataset['train'])
    for i in range(size_squad):
        psg_text = []
        psg_emd = []
        question_list.append(squad_dataset['train'][i]['question'])
        psg_text.append(squad_dataset['train'][i]['context'])
        neg_psg_ids = create_neg_ids(pos_id=i, num_data= size_squad, num_gen= 10)
        for j in neg_psg_ids:
            psg_text.append(squad_dataset['train'][j]['context'])
        for text in psg_text:
            input_ids = tokenizer(text,return_tensors="pt")['input_ids']
            emd = opt_model(input_ids = input_ids)['last_hidden_state'].detach()[0][-1]
            psg_emd.append(emd)
        psg_emd = torch.stack(psg_emd)
        context_list.append(psg_emd)
    return {"question": question_list, "context": context_list}

def convert_dic_json(dic:dict,path):
    with open(path, "w") as outfile:
        json.dump(dic, outfile)

def main_squad_json(path = "data/json_data/sample.json"):
    dic = convert_squad_dic()
    convert_dic_json(dic,path)

def load_json_dataset(path):
    temp_dataset = load_dataset("json", data_files=path, split="train")
    return temp_dataset


if __name__ == "__main__":
    main_squad_json(path = "data/json_data/sample.json")


