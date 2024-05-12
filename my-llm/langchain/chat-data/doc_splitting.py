

#导入文本分割器
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter



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
        

    


if __name__ == '__main__':
    # print("Hello, World!")
    docSplitter = DocSplitting()
    # docSplitter.recursive_character_text_splitter()
    docSplitter.character_text_splitter()

























