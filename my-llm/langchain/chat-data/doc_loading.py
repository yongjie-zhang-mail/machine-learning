

# 下载最新版 LangChain
# pip install -q langchain --upgrade
# 安装第三方库 pypdf
# pip install -q pypdf
# youtube 相关包
# pip -q install yt_dlp
# pip -q install pydub


# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader

# from langchain.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

# from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader

import json

# from langchain.document_loaders import NotionDirectoryLoader
from langchain_community.document_loaders import NotionDirectoryLoader



class DocLoading:

    def __init__(self):
        self.webpage_url = "https://github.com/datawhalechina/d2l-ai-solutions-manual/blob/master/docs/README.md"
        
    def view_docs(self, pages):
        print("Type of pages: ", type(pages))
        print("Length of pages: ", len(pages))

        page = pages[0]
        print("Type of page: ", type(page))
        print("Page_content: ", page.page_content[:500])
        print("Meta Data: ", page.metadata)
        return page
    
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

    def load_youtube(self):
        url="https://www.youtube.com/watch?v=jGwO_UgTS7I"
        save_dir="docs/youtube/"

        # 创建一个 GenericLoader Class 实例
        loader = GenericLoader(
            #将链接url中的Youtube视频的音频下载下来,存在本地路径save_dir
            YoutubeAudioLoader([url],save_dir), 
            
            #使用OpenAIWhisperPaser解析器将音频转化为文本
            OpenAIWhisperParser()
        )

        # 调用 GenericLoader Class 的函数 load对视频的音频文件进行加载
        pages = loader.load()

        page = self.view_docs(pages)

    def load_webpage(self): 
        # 创建一个 WebBaseLoader Class 实例
        url = self.webpage_url
        header = {'User-Agent': 'python-requests/2.27.1', 
                'Accept-Encoding': 'gzip, deflate, br', 
                'Accept': '*/*',
                'Connection': 'keep-alive'}
        loader = WebBaseLoader(web_path=url,header_template=header)

        # 调用 WebBaseLoader Class 的函数 load对文件进行加载
        pages = loader.load()

        page = self.view_docs(pages)

        # convert_to_json = json.loads(page.page_content)
        # extracted_markdow = convert_to_json['payload']['blob']['richText']
        # print(extracted_markdow)

    def load_notion(self): 
        loader = NotionDirectoryLoader("docs/Notion_DB")
        pages = loader.load()

        page = self.view_docs(pages)
    


if __name__ == '__main__':
    # print("Hello, World!")
    docLoad = DocLoading()
    # docLoad.load_doc()
    docLoad.load_webpage()
    # docLoad.load_notion()























