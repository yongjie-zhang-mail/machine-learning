
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

class DocLoading:

    def __init__(self):
        self.list_url = 'https://www.ncbi.nlm.nih.gov/omim'

    def estrip(self, string):
        if string:
            return string.strip()
        else:
            return None
    
    def load_doc(self): 
        # 创建一个 PyPDFLoader Class 实例，输入为待加载的pdf文档路径
        loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
        # loader = PyPDFLoader("docs/matplotlib/第一回：Matplotlib初相识.pdf")

        # 调用 PyPDFLoader Class 的函数 load对pdf文件进行加载
        pages = loader.load()

        # debug print
        print(f'pages 的类型是： \n {type(pages)}. \n')
        print(f'pages 的长度是： \n {len(pages)}. \n')

        page = pages[0]
        print(f'page 的类型是： \n {type(page)}. \n')
        print(f'第一个page对象的内容，前500个字符是： \n {page.page_content[0:500]}. \n')
        print(f'page 的元信息是： \n {page.metadata}. \n')





if __name__ == '__main__':
    # print("Hello, World!")
    docLoad = DocLoading()
    docLoad.load_doc()


    



















