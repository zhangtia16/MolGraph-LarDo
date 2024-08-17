# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
import time

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors


from transformers import AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# 计算分子量
def calculate_molecular_weight(smiles, is_round=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if is_round:
        return round(Descriptors.MolWt(mol), 1)
    else:
        return Descriptors.MolWt(mol)

# 计算分配系数（logP）
def calculate_logp(smiles, is_round=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    if is_round:
        return round(Crippen.MolLogP(mol), 3)
    else:
        return Crippen.MolLogP(mol)



import csv 
def read_smiles(data_path):
    smiles2text = {}
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):  

            if True: #
                smiles = row['smiles'] 
                smiles2text[smiles] = ""
                # mol = Chem.MolFromSmiles(smiles)
                # if mol != None:
                #     smiles_data.append(smiles)
    return smiles2text

def return_prompt(molecule_smiles,version):

    if version == 'wo_stage1':
        user_prompt = "<MOLECULE>: {}\n".format(molecule_smiles)   
        user_prompt += "Provide the concise information about <MOLECULE>:\n"
        user_prompt += "<end of prompt>"

    elif version == 'wo_rdkit':
        user_prompt = "<PROPERTIES>: Lipophilicity; Molecular weight; Hydrogen bond donors and acceptors; Metabolic stability\n"
        user_prompt += "<MOLECULE>: {}\n".format(molecule_smiles)   
        user_prompt += "Provide the concise information about <PROPERTIES> of <MOLECULE> in the following format:\n"
        user_prompt += "[Lipophilicity]: ...\n[Molecular weight]: ...\n[Hydrogen bond donors and acceptors]...\n[Metabolic stability]: ..."
        user_prompt += "<end of prompt>"

    elif version == 'prompt_rdkit':
        user_prompt = "<PROPERTIES>: Lipophilicity; Molecular weight; Hydrogen bond donors and acceptors; Metabolic stability\n"
        user_prompt += "<MOLECULE>: {}\n".format(molecule_smiles)   
        user_prompt += "LogP of <MOLECULE>: {}\n".format(calculate_logp(molecule_smiles))  
        user_prompt += "Molecular weight of <MOLECULE>: {} g/mol\n".format(calculate_molecular_weight(molecule_smiles))  
        user_prompt += "Provide the concise information about <PROPERTIES> of <MOLECULE> in the following format:\n"
        user_prompt += "[Lipophilicity]: ...\n[Molecular weight]: ...\n[Hydrogen bond donors and acceptors]...\n[Metabolic stability]: ..."
        user_prompt += "<end of prompt>"
    else:
        pass

    return user_prompt


def extract_information(input_str):
    keyword = "<end of prompt>"
    index = input_str.find(keyword)
    if index != -1:
        return input_str[index + len(keyword):]
    else:
        return "Keyword not found in the input string."


def main(version='wo_rdkit'):
  

    system_prompt = "You are a biochemistry expert.\n"   
    
    dataset = 'bbbp'
    data_path = '/home/zty/codes/MolCLR-master-modify/data/{}/BBBP.csv'.format(dataset)

    output_path = '/home/zty/codes/MolCLR-master-modify/md_text/{}/{}_{}_response.csv'.format(version, dataset, 'mistral')
    print(output_path)
    
    
    with open(output_path, 'w', encoding='utf-8') as f:  
        f.write('smiles\tresponse\tversion:Mistral-7B-Instruct-v0.2\n')


    smiles2text = read_smiles(data_path)    # 2050
    sample_num = len(smiles2text)

    start_time = time.time()
    


    # llm
    model_name = "/workspace1/zty/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )
    device = "cuda"


    for i, smiles in enumerate(smiles2text.keys()): 

 
        
        user_prompt = return_prompt(smiles,version)
        
        encodeds = tokenizer(user_prompt, return_tensors="pt",add_special_tokens=True)
        model_inputs = encodeds.to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=200, do_sample=True)
        # decode with mistral tokenizer
        result = tokenizer.decode(generated_ids[0].tolist())

        result = extract_information(result)

        with open(output_path, 'a', encoding='utf-8') as f:
            f.write('{}\t{}\n'.format(smiles, result.replace('\n', ' ').replace(smiles, '<MOLECULE>').replace('The molecule', '<MOLECULE>').replace('the molecule', '<MOLECULE>')))
            # f.close()
            
        cur_time = time.time()
        print('Genertaion Process: {}/{} Duration: {:.2f} seconds'.format(i,sample_num,0))
        



if __name__ == "__main__":

    main()


