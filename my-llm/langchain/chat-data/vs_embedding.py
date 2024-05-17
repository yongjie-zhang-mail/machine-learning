
# pip install -Uq pypdf
# pip install -Uq chromadb

# 通过 modelscope 下载模型
# pip install modelscope==1.9.5
# pip install transformers==4.35.2
# pip install modelscope
# pip install transformers



import os
from modelscope.hub.snapshot_download import snapshot_download

# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

class VsEmbedding:

    def __init__(self):
        self.text2 = "abcdefghijklmnopqrstuvwxyzabcdefg"


    def download_model(self):
        # 创建保存模型目录
        os.system("mkdir /root/models")

        # save_dir是模型保存到本地的目录
        save_dir="/root/models"

        snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                        cache_dir=save_dir, 
                        revision='v1.1.0')    

    
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
        embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")
        embeddings.embed_query



if __name__ == '__main__':
    # print("Hello, World!")
    vsEmbedding = VsEmbedding()
    vsEmbedding.load_doc()

























