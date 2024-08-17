# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional
from transformers import AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def return_prompt(dataset):

    if dataset == 'bace':
        user_prompt = "<DATASET>: BACE\n"
        user_prompt += "Description of <DATASET>: The BACE dataset provides quantitative IC50 and qualitative (binary label) binding results for a set of inhibitors of human beta-secretase 1 (BACE-1).\n"
        user_prompt += "<TASK>: Classification\n"
        user_prompt += "Target of <TASK>: Binding results (binary label) for a set of inhibitors of human beta-secretase 1 (BACE-1).\n"

    elif dataset == 'freesolv':
        user_prompt = "<DATASET>: FreeSolv\n"
        user_prompt += "Description of <DATASET>: The FreeSolv dataset is a collection of experimental and calculated hydration free energies for small molecules in water, along with their experiemental values. Here, we are using a modified version of the dataset with the molecule smile string and the corresponding experimental hydration free energies.\n"
        user_prompt += "<TASK>: Regression\n"
        user_prompt += "Target of <TASK>: Experimental hydration free energy.\n"
    
    elif dataset == 'esol':
        user_prompt = "<DATASET>: ESOL\n"
        user_prompt += "Description of <DATASET>: The Delaney (ESOL) dataset a regression dataset containing structures and water solubility data for 1128 compounds. The dataset is widely used to validate machine learning models on estimating solubility directly from molecular structures (as encoded in SMILES strings).\n"
        user_prompt += "<TASK>: Regression\n"
        user_prompt += "Target of <TASK>: Log-scale water solubility\n"
    
    elif dataset == 'lipo':
        user_prompt = "<DATASET>: Lipophilicity\n"
        user_prompt += "Description of <DATASET>: Lipophilicity is an important feature of drug molecules that affects both membrane permeability and solubility. The lipophilicity dataset, curated from ChEMBL database, provides experimental results of octanol/water distribution coefficient (logD at pH 7.4) of 4200 compounds.\n"
        user_prompt += "<TASK>: Regression\n"
        user_prompt += "Target of <TASK>: Measured octanol/water distribution coefficient (logD) of the compound\n"

    elif dataset == 'sider':
        user_prompt = "<DATASET>: SIDER\n"
        user_prompt += "Description of <DATASET>: The Side Effect Resource (SIDER) is a database of marketed drugs and adverse drug reactions (ADR). The version of the SIDER dataset in DeepChem has grouped drug side effects into 27 system organ classes following MedDRA classifications measured for 1427 approved drugs.\n"
        user_prompt += "<TASK>: Classification\n"
        user_prompt += "Target of <TASK>: 27 system organ classes following MedDRA classifications\n"
    
    elif dataset == 'tox21':
        user_prompt = "<DATASET>: Tox21\n"
        user_prompt += "Description of <DATASET>: The 'Toxicology in the 21st Century' (Tox21) initiative created a public database measuring toxicity of compounds, which has been used in the 2014 Tox21 Data Challenge. This dataset contains qualitative toxicity measurements for 8k compounds on 12 different targets, including nuclear receptors and stress response pathways.\n"
        user_prompt += "<TASK>: Classification\n"
        user_prompt += "Target of <TASK>: Nuclear receptor signaling bioassays results and Stress response bioassays results\n"
    
    elif dataset == 'hiv':
        user_prompt = "<DATASET>: HIV\n"
        user_prompt += "Description of <DATASET>: The HIV dataset was introduced by the Drug Therapeutics Program (DTP) AIDS Antiviral Screen, which tested the ability to inhibit HIV replication for over 40,000 compounds. Screening results were evaluated and placed into three categories: confirmed inactive (CI),confirmed active (CA) and confirmed moderately active (CM). We further combine the latter two labels, making it a classification task between inactive (CI) and active (CA and CM).\n"
        user_prompt += "<TASK>: Classification\n"
        user_prompt += "Target of <TASK>: 'HIV_active': Binary labels for screening results: 1 (CA/CM) and 0 (CI)\n"

    
    elif dataset == 'qm7':
        user_prompt = "<DATASET>: QM7\n"
        user_prompt += "Description of <DATASET>: QM7 is a subset of GDB-13 (a database of nearly 1 billion stable and synthetically accessible organic molecules) containing up to 7 heavy atoms C, N, O, and S. The 3D Cartesian coordinates of the most stable conformations and their atomization energies were determined using ab-initio density functional theory (PBE0/tier2 basis set).\n"
        user_prompt += "<TASK>: Regression\n"
        user_prompt += "Target of <TASK>: 'u0_atom': atomization energies\n"
    

    user_prompt += "Given the SMILES of a <Molecule>, what are the are the molecular <PROPERTIES> related to <TASK>?"

    return user_prompt

  

def main():
    system_prompt = "You are a biochemistry expert.\n"   
    

    output_path = '/home/zty/codes/MolCLR-master-modify/md_text/{}_{}_stage1_properties_qm7.csv'.format('reg', 'mistral')

    print(output_path)



    # llm
    model_name = "/workspace1/zty/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )
    device = "cuda"


    for dataset in ['qm7']:
        user_prompt = system_prompt + return_prompt(dataset)
        
        encodeds = tokenizer(user_prompt, return_tensors="pt")
        model_inputs = encodeds.to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=300, do_sample=True)
        # decode with mistral tokenizer
        result = tokenizer.decode(generated_ids[0].tolist())

        with open(output_path, 'a', encoding='utf-8') as f:
            f.write('{}\t{}\n'.format(dataset, result))
            # f.close()
            
        



if __name__ == "__main__":

    main()


