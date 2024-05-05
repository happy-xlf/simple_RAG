#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   simple_demo.py
@Time    :   2024/05/05 14:08:42
@Author  :   Lifeng
@Version :   1.0
@Desc    :   None
'''

import os
import argparse
import faiss
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
from tqdm import tqdm

# 文件加载
def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        sentence = text.split('。')
    return text, sentence

# 构建prompt
def generate_rag_prompt(data_point):
    return f"""你是一个根据参考背景来回答用户问题的专家，需要保证回答的内容来自于参考背景，回答简洁，准确。
### 问题:
{data_point['instruction']}
### 参考背景:
{data_point['input']}
### 回答:
    """

# Embedding类
class DocumentEmbedding:

    def __init__(self, model_name, max_length, max_number_of_sentences):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.max_number_of_sentences = max_number_of_sentences

    def get_document_embeddings(self, sentences):
        sentences = sentences[:self.max_number_of_sentences]

        encoder_input = self.tokenizer(sentences,
                                       padding=True,
                                       truncation=True, 
                                       max_length=self.max_length,
                                       return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**encoder_input)

        return torch.mean(model_output.pooler_output, dim=0, keepdim=True)
    
# 生成模型
class GenerativeModel:
    def __init__(self, model_path, max_input_length=128, max_generated_length=200):
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                                          trust_remote_code=True,
                                                          torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
            add_eos_token=True,
            add_bos_token=True,
            use_fast=False
            )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_length = max_input_length
        self.max_generated_length = max_generated_length
        self.device = torch.device("cuda:4") if torch.cuda.is_available() else torch.device("cpu")
        
        self.model.to(self.device)

    def answer_prompt(self, prompt):
        encoder_input = self.tokenizer([prompt],
                                       padding="max_length",
                                       max_length=self.max_input_length,
                                       return_tensors="pt"
                                       )
        outputs = self.model.generate(
            input_ids = encoder_input['input_ids'].to(self.device),
            attention_mask = encoder_input['attention_mask'].to(self.device),
            max_new_tokens=self.max_generated_length
        )

        decoder_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return decoder_text
    
if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--documents_directory",
                       help="The directory containing the documents.",
                       default="rag_documents")
    parse.add_argument("--embedding_model",
                       help="The embedding model to use.",
                       default="/home/xulifeng/Models/bge-large-zh-v1.5")
    parse.add_argument("--generative_model",
                       help="The generative model to use.",
                       default="/data/share/models/Qwen1.5-1.8B-Chat")
    parse.add_argument("--number_of_docs",
                       help="The number of documents to use.",
                       default=5)
    args = parse.parse_args()

    print("Splitting documents into sentences...")
    documents={}
    idx=0
    for k,file in enumerate(tqdm(os.listdir(args.documents_directory)[:10])):
        current_filepath = os.path.join(args.documents_directory, file)
        text, sentence = process_file(current_filepath)
        sentence = [s for s in sentence if len(s) > 0]
        for s in sentence:
            documents[idx] = {'file_path':file,'document_text':[s]}
            idx+=1
        # documents[idx] = {'file_path':file,
        #                   'sentences':sentence,
        #                   'document_text':text}
    
        
    print("Building document embeddings...")
    document_embedder = DocumentEmbedding(model_name=args.embedding_model,max_length=512,max_number_of_sentences=20)
    embeddings = []

    for idx in tqdm(documents):
        embeddings.append(document_embedder.get_document_embeddings(documents[idx]['document_text']))
    embeddings = torch.concat(embeddings, dim=0).data.cpu().numpy()
    embedding_dimensions = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(int(embedding_dimensions))
    faiss_index.add(embeddings)

    
    generativate_model = GenerativeModel(model_path=args.generative_model,
                                        max_input_length=512,
                                        max_generated_length=256)
    
    question = input("请输入问题：")
    while question!="q":
        # question = "天津旅游推荐？"
        query_embedding = document_embedder.get_document_embeddings([question])
        distances, indices = faiss_index.search(query_embedding.data.cpu().numpy(), k=int(args.number_of_docs))


        context = ""
        for idx in indices[0]:
            context += documents[idx]['document_text'][0]
        
        rag_prompt = generate_rag_prompt({"instruction":question, "input":context})

        print("Generating answer...")
        ori_answer = generativate_model.answer_prompt(rag_prompt)[0]
        print(ori_answer)
        print("-----------------------------")
        answer = ori_answer.split("### 回答:")[1]
        print(answer)
        question = input("请输入问题：")