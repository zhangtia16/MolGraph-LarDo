from rdkit import Chem  
from rdkit import DataStructs

def calculate_similarity(smiles1, smiles2):  
    mol1 = Chem.MolFromSmiles(smiles1)  
    mol2 = Chem.MolFromSmiles(smiles2)  

    fp1 = Chem.RDKFingerprint(mol1)
    fp2 = Chem.RDKFingerprint(mol2)

    similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
    return similarity

def find_most_similar_sample(smiles_list, target_index):  
    
    similarity_matrix = []  
    for i in range(len(smiles_list)):  
        
        if i != target_index:  
            similarity_matrix.append(calculate_similarity(smiles_list[target_index], smiles_list[i]))
            
            
    max_index = similarity_matrix.index(max(similarity_matrix))  
    return max_index

def pos_sampling(smiles_list):  

    print('Sampling ....')
    result = {}  
    for i in range(len(smiles_list)):  
        if i % 100 ==0:
            print('{}/{}'.format(i,len(smiles_list)))
        target_index = i  
        most_similar_index = find_most_similar_sample(smiles_list, target_index)  
        result[i] = most_similar_index  
    return result



