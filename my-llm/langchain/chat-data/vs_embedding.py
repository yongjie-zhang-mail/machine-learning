
# pip install -Uq pypdf
# pip install -Uq chromadb

# 通过 modelscope 下载模型
# pip install modelscope==1.9.5
# pip install transformers==4.35.2
# pip install modelscope
# pip install transformers

# mkdir -p /lab/models
# pip install -U sentence-transformers

# 若使用 root 账号启动
# jupyter notebook --allow-root


import os
# from modelscope.hub.snapshot_download import snapshot_download
from modelscope import snapshot_download

# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer

import numpy as np


class VsEmbedding:

    def __init__(self):
        self.text2 = "abcdefghijklmnopqrstuvwxyzabcdefg"


    def download_embedding_model(self):
        # save_dir是模型保存到本地的目录
        save_dir="/lab/models"
        # 模型下载
        # snapshot_download("xrunda/m3e-base", 
        #                 cache_dir=save_dir, 
        #                 revision='v1.1.0')
        snapshot_download("xrunda/m3e-base", 
                        cache_dir=save_dir)


    def test_m3e(self): 
        # /lab/models/m3e-base
        # model = SentenceTransformer('moka-ai/m3e-base')
        model = SentenceTransformer('/lab/models/m3e-base')

        #Our sentences we like to encode
        sentences = [
            '* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem',
            '* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练',
            '* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one'
        ]

        #Sentences are encoded by calling model.encode()
        embeddings = model.encode(sentences)

        #Print the embeddings
        for sentence, embedding in zip(sentences, embeddings):
            print("Sentence:", sentence)
            print("Embedding:", embedding)
            print("")


    def load_doc(self): 
        # 加载 PDF
        loaders = [
            # 故意添加重复文档，使数据混乱
            PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
            PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf"),
            PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture02.pdf"),
            PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture03.pdf")
        ]
        docs = []
        for loader in loaders:
            docs.extend(loader.load())
   
        # 分割文本
        text_splitter = RecursiveCharacterTextSplitter(
            # 每个文本块的大小。这意味着每次切分文本时，会尽量使每个块包含 1500 个字符。
            chunk_size = 1500,  
            # 每个文本块之间的重叠部分。
            chunk_overlap = 150  
        )

        splits = text_splitter.split_documents(docs)
        print(len(splits))

    def embedding(self):
        # sentence1 = "i like dogs"
        # sentence2 = "i like canines"
        # sentence3 = "the weather is ugly outside"

        sentence1 = "我喜欢狗"
        sentence2 = "我喜欢犬科动物"
        sentence3 = "外面的天气很糟糕"

        # /lab/models/m3e-base        
        embedding_model = SentenceTransformer('/lab/models/m3e-base')

        #Sentences are encoded by calling model.encode()
        embedding1 = embedding_model.encode(sentence1)
        # print("Embedding1:", embedding1)
        # 计算嵌入向量的模
        # print(np.linalg.norm(embedding1))
        embedding1 = self.norm(embedding1)
        # print(np.linalg.norm(embedding1))

        embedding2 = self.norm(embedding_model.encode(sentence2))
        embedding3 = self.norm(embedding_model.encode(sentence3))

        print(np.dot(embedding1, embedding2))
        print(np.dot(embedding1, embedding3))
        print(np.dot(embedding2, embedding3))


    def norm(self, embedding):
        return embedding / np.linalg.norm(embedding)


if __name__ == '__main__':
    # print("Hello, World!")
    vsEmbedding = VsEmbedding()
    # vsEmbedding.download_embedding_model()
    # vsEmbedding.test_m3e()
    vsEmbedding.embedding()

























