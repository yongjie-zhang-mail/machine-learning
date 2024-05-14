
# pip install -Uq pypdf
# pip install -Uq chromadb

# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter



class VsEmbedding:

    def __init__(self):
        self.text2 = "abcdefghijklmnopqrstuvwxyzabcdefg"
        

    
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



if __name__ == '__main__':
    # print("Hello, World!")
    vsEmbedding = VsEmbedding()
    vsEmbedding.load_doc()

























