from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors

import csv

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
    

# 计算氢键供体和受体数量
def calculate_hydrogen_bond_donors_acceptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    # 计算氢键供体数量
    num_hbd = rdMolDescriptors.CalcNumHBD(mol)
    
    # 计算氢键受体数量
    num_hba = rdMolDescriptors.CalcNumHBA(mol)
    
    return num_hbd, num_hba


# 示例函数
def test():
    # 示例化合物的SMILES表示
    smiles_example = 'C(=O)(OC(C)(C)C)CCCc1ccc(cc1)N(CCCl)CCCl'  

    # 计算分子量
    molecular_weight = calculate_molecular_weight(smiles_example)

    logp_value = calculate_logp(smiles_example)

    num_hydrogen_bond_donors, num_hydrogen_bond_acceptors = calculate_hydrogen_bond_donors_acceptors(smiles_example)


def substitue_content(llm_response, smiles, dataset):

    def keep_after_first_sign(input_string, sign):
        # 寻找第一个 "SIGN" 的位置
        sign_index = input_string.find(sign)
        
        if sign_index != -1:  # 如果找到了 "SIGN"
            # 返回从 "SIGN" 开始到字符串末尾的子串
            result_string = input_string[sign_index:]
        else:
            # 如果未找到 "SIGN"，则返回空字符串
            result_string = ""
    
        return result_string

    if dataset == 'bbbp':
        molecular_weight = calculate_molecular_weight(smiles)
        logp_value = calculate_logp(smiles)

        new_llm_response = "[Lipophilicity]:{}. [Molecular weight]:{}. {}".format(logp_value, molecular_weight, keep_after_first_sign(llm_response, sign='[Hydrogen bond donors and acceptors]:'))


    return new_llm_response



    return new_llm_response

def rdkit_process(llm_version, dataset):
    if llm_version == 'llama2':
        csv_path = '/workspace1/zty/llama-main/llama2_response/{}_{}_response.csv'.format(dataset, llm_version)
    else:
        csv_path = './gpt_response/{}_{}_response.csv'.format(dataset, llm_version)
    
    output_path = './md_text/version1/{}_{}_response.csv'.format(dataset, llm_version)


    
    with open(output_path, 'w', encoding='utf-8') as f:  
        f.write('smiles\tresponse\tversion:llama2-7b-chat\n')
    

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        next(csv_reader)
        for row in csv_reader:
            smiles = row[0]
            llm_response = row[1]
            
            new_llm_response = substitue_content(llm_response, smiles, dataset)

            with open(output_path, 'a', encoding='utf-8') as f:
                f.write('{}\t{}\n'.format(smiles, new_llm_response))



llm_version = 'llama2'
for dataset in ['bbbp']: #,'clintox']:#['bbbp','clintox','bace','sider','tox21','hiv','muv']:
    rdkit_process(llm_version, dataset)