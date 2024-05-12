

#导入文本分割器
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

from langchain_community.document_loaders import NotionDirectoryLoader

from langchain.text_splitter import TokenTextSplitter

from langchain.text_splitter import MarkdownHeaderTextSplitter


class DocSplitting:

    def __init__(self):
        
        # 设置块大小
        self.chunk_size = 26 
        # 设置块重叠大小
        self.chunk_overlap = 4 

        self.text2 = "abcdefghijklmnopqrstuvwxyzabcdefg"
        self.text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"


    def view_results(self, input, output, splitter):
        print(f'文本：{input} \n'
              f'切分器： {splitter} \n'               
              f'切分结果：{output} \n')

        
    def recursive_character_text_splitter(self):        
        # 初始化文本分割器
        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        splitter = f'RecursiveCharacterTextSplitter(chunk_size={r_splitter._chunk_size},chunk_overlap={r_splitter._chunk_overlap})'

        # 使用递归字符文本分割器                
        r_splits2 = r_splitter.split_text(self.text2)                
        self.view_results(input=self.text2, output=r_splits2, splitter=splitter)
        
        r_splits3 = r_splitter.split_text(self.text3)
        self.view_results(input=self.text3, output=r_splits3, splitter=splitter)


    def character_text_splitter(self): 
        c_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        splitter1 = f'CharacterTextSplitter(chunk_size={c_splitter._chunk_size},chunk_overlap={c_splitter._chunk_overlap})'

        # 字符文本分割器
        c_splits3 = c_splitter.split_text(self.text3)
        # 可以看到字符分割器没有分割这个文本，因为字符文本分割器默认以换行符为分隔符
        self.view_results(input=self.text3, output=c_splits3, splitter=splitter1)
        
        # 设置空格分隔符
        c_splitter2 = CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separator=' '
        )
        splitter2 = f'CharacterTextSplitter(chunk_size={c_splitter._chunk_size},chunk_overlap={c_splitter._chunk_overlap},separator="{c_splitter2._separator}")'

        c_splits3_2 = c_splitter2.split_text(self.text3)
        self.view_results(input=self.text3, output=c_splits3_2, splitter=splitter2)
        

    def a1(self): 
        # 递归分割长段落
        some_text1 = """When writing documents, writers will use document structure to group content. \
        This can convey to the reader, which idea's are related. For example, closely related ideas \
        are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
        Paragraphs are often delimited with a carriage return or two carriage returns. \
        Carriage returns are the "backslash n" you see embedded in this string. \
        Sentences have a period at the end, but also, have a space.\
        and words are separated by space."""
        
        print(len(some_text1))
        
        # 中文版
        some_text2 = """在编写文档时，作者将使用文档结构对内容进行分组。 \
            这可以向读者传达哪些想法是相关的。 例如，密切相关的想法\
            是在句子中。 类似的想法在段落中。 段落构成文档。 \n\n\
            段落通常用一个或两个回车符分隔。 \
            回车符是您在该字符串中看到的嵌入的“反斜杠 n”。 \
            句子末尾有一个句号，但也有一个空格。\
            并且单词之间用空格分隔"""

        print(len(some_text2))

        ''' 
        依次传入分隔符列表，分别是双换行符、单换行符、空格、空字符，
        因此在分割文本时，首先会采用双分换行符进行分割，同时依次使用其他分隔符进行分割
        '''

        c_splitter = CharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=0,
            separator=' '
        )
        c_splits = c_splitter.split_text(some_text1)
        print(c_splits)


        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=0,
            separators=["\n\n", "\n", " ", ""]
        )
        r_splits = r_splitter.split_text(some_text1)
        print(r_splits)
        print(r_splitter.split_text(some_text2))



        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=0,
            separators=["\n\n", "\n", "\. ", " ", ""]
        )
        r_splits = r_splitter.split_text(some_text1)
        print(r_splits)

        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=0,
            separators=["\n\n", "\n", "?<=\. ", " ", ""]
        )
        r_splits = r_splitter.split_text(some_text1)
        print(r_splits)
        print(r_splitter.split_text(some_text2))

        # 这就是递归字符文本分割器名字中“递归”的含义，总的来说，我们更建议在通用文本中使用递归字符文本分割器


    def a2(self):
        loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
        pages = loader.load()

        text_splitter = CharacterTextSplitter(                        
            separator="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        docs = text_splitter.split_documents(pages)
        print(len(docs))
        print(len(pages))


    def a3(self):
        loader = NotionDirectoryLoader("docs/Notion_DB")
        notion_db = loader.load()

        text_splitter = CharacterTextSplitter(                        
            separator="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )

        docs = text_splitter.split_documents(notion_db)
        print(len(notion_db))
        print(len(docs))


    def a4(self):
        # 使用token分割器进行分割，
        # 将块大小设为1，块重叠大小设为0，相当于将任意字符串分割成了单个Token组成的列
        text_splitter = TokenTextSplitter(chunk_size=1, chunk_overlap=0)
        text1 = "foo bar bazzyfoo"
        print(text_splitter.split_text(text1))

        loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
        pages = loader.load()

        text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        print(docs[0])
        print(pages[0].metadata)


    def a5(self):
        # 定义一个Markdown文档
        markdown_document = """# Title\n\n \
        ## Chapter 1\n\n \
        Hi this is Jim\n\n Hi this is Joe\n\n \
        ### Section \n\n \
        Hi this is Lance \n\n 
        ## Chapter 2\n\n \
        Hi this is Molly"""

        # 定义想要分割的标题列表和名称
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        # 初始化Markdown标题文本分割器，分割Markdown文档
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        md_header_splits = markdown_splitter.split_text(markdown_document)

        print(md_header_splits[0])
        print(md_header_splits[1])

    def a6(self):
        #加载数据库的内容
        loader = NotionDirectoryLoader("docs/Notion_DB")
        docs = loader.load()
        #拼接文档
        txt = ' '.join([d.page_content for d in docs])

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        #加载文档分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )

        #分割文本内容
        md_header_splits = markdown_splitter.split_text(txt)

        #分割结果
        print(md_header_splits[0])









if __name__ == '__main__':
    # print("Hello, World!")
    docSplitter = DocSplitting()
    # docSplitter.recursive_character_text_splitter()
    # docSplitter.character_text_splitter()
    # docSplitter.a1()
    # docSplitter.a2()
    # docSplitter.a3()
    # docSplitter.a4()
    # docSplitter.a5()
    docSplitter.a6()


























