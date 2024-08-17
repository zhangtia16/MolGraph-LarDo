from sentence_transformers import SentenceTransformer
import pickle
import csv
import numpy as np


def get_smiles2text(csv_path):
    smiles2text = {}

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        next(csv_reader)
        for row in csv_reader:
            smiles = row[0]
            gpt_response = row[1]
            smiles2text[smiles] = gpt_response
    
    return smiles2text


import re

def remove_brackets_content(input_string):
    # 使用正则表达式去除方括号内的内容
    result_string = re.sub(r'\[[^\]]*\]', '', input_string).replace(':',';').lstrip(';')
    return result_string

def bert_encode_numpy(smiles2text):
    sample_num = len(smiles2text.keys())
    embeddings = []
    model = SentenceTransformer('/home/zty/huggingface_models/all-mpnet-base-v2')
    for i, smiles in enumerate(smiles2text.keys()):  
        text = smiles2text[smiles]
        embedding = model.encode(text)
        embeddings.append(embedding)

         # get rid of template
        # text = remove_brackets_content(text)
        
        if i % 500 == 0:
            print('Encoded Samples: {}/{}'.format(i, sample_num))
    embeddings = np.array(embeddings)
    print('embeddings.shape:', embeddings.shape)

    return embeddings




def bert_encode_dict(smiles2text):
    sample_num = len(smiles2text.keys())
    smiles2embeddings = {}
    model = SentenceTransformer('/home/zty/huggingface_models/all-mpnet-base-v2')
    for i, smiles in enumerate(smiles2text.keys()):  
        text = smiles2text[smiles]

        # get rid of template
        # text = remove_brackets_content(text)



        embedding = model.encode(text)
        smiles2embeddings[smiles] = embedding
        
        if i % 500 == 0:
            print('Encoded Samples: {}/{}'.format(i, sample_num))
            # print(text)

    return smiles2embeddings

from transformers import AutoModel, AutoTokenizer


def bert_encode_numpy_scibert(smiles2text):
    sample_num = len(smiles2text.keys())
    embeddings = []
    text_tokenizer = AutoTokenizer.from_pretrained('/workspace1/zty/scibert')
    text_model = AutoModel.from_pretrained('/workspace1/zty/scibert')

    for i, smiles in enumerate(smiles2text.keys()):  
        text = smiles2text[smiles]

        # get rid of template
        # text = remove_brackets_content(text)

        inputs = text_tokenizer(text, return_tensors="pt")
        outputs = text_model(**inputs)
    
        embedding = outputs["pooler_output"]
        embeddings.append(embedding.detach().numpy())
        if i % 100 == 0:
            print('Encoded Samples: {}/{}'.format(i, sample_num))
    embeddings = np.array(embeddings)
    print('embeddings.shape:', embeddings.shape)

    return embeddings

def bert_encode_dict_scibert(smiles2text):
    sample_num = len(smiles2text.keys())
    smiles2embeddings = {}
    text_tokenizer = AutoTokenizer.from_pretrained('/workspace1/zty/scibert')
    text_model = AutoModel.from_pretrained('/workspace1/zty/scibert')

    for i, smiles in enumerate(smiles2text.keys()):  
        text = smiles2text[smiles]

        # get rid of template
        # text = remove_brackets_content(text)

        inputs = text_tokenizer(text, return_tensors="pt")
        outputs = text_model(**inputs)
    
        embedding = outputs["pooler_output"]
        smiles2embeddings[smiles] = embedding
        
        if i % 500 == 0:
            print('Encoded Samples: {}/{}'.format(i, sample_num))
            # print(text)

    return smiles2embeddings


def save_to_pkl(embeddings, pkl_path):
    with open(pkl_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print('Text embeddings saved to {}'.format(pkl_path))


llm = 'mistral' #'llama2' 

version = 'wo_rdkit'
for dataset in ['bbbp','bace','sider','freesolv','esol','lipo','qm7']: #,'clintox']:#['bbbp','clintox','bace','sider','tox21','hiv']:
    print(dataset)
    # if llm == 'llama2' or llm == 'mistral':
    #     #csv_path = '/workspace1/zty/llama-main/llama2_response/{}_{}_response.csv'.format(dataset, llm)
    #     csv_path = './md_text/prompt_rdkit/{}_{}_response.csv'.format(dataset, llm)
    # else:
    #     csv_path = './gpt_response/{}_{}_response.csv'.format(dataset, llm)

    
    csv_path = './md_text/{}/{}_{}_response.csv'.format(version,dataset, llm)

    smiles2text = get_smiles2text(csv_path)

    pkl_path = './text_embeddings/{}/{}_{}_embeddings.pkl'.format(version,dataset, llm) #_remove_template
    embeddings = bert_encode_dict(smiles2text)
    save_to_pkl(embeddings, pkl_path)


    # pkl_path = './text_embeddings/{}_{}_embeddings_scibert.pkl'.format(dataset, llm)  # _remove_template
    # embeddings = bert_encode_numpy_scibert(smiles2text)
    # save_to_pkl(embeddings, pkl_path)

