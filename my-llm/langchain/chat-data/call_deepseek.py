
# pip install langchain_openai



import os
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class CallDeepSeek:

    def __init__(self):
        self.deepseek_key = "sk-089ab2267dfb4cdbb1ec50e6aed5e98c"


    def test(self):
        
        llm = ChatOpenAI(
            temperature=0.95,
            model="deepseek-chat",
            # api_key=self.deepseek_key,
            # base_url="https://api.deepseek.com"  
            openai_api_key =self.deepseek_key,
            openai_api_base="https://api.deepseek.com"            
        )

        prompt = ChatPromptTemplate.from_template("请根据下面的主体写一篇小红书营销的短文： {topic}")
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        chain.invoke({"topic": "康师傅绿茶"})
        
        # output_parser = StrOutputParser()
        # chain = llm | output_parser
        # chain.invoke("你好")



    def test2(self): 
        client = OpenAI(api_key=self.deepseek_key, base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                # {"role": "user", "content": "Hello"},
                {"role": "user", "content": "对氯间二甲苯酚 和 洗衣液，在洗衣服时 能同时放吗？"},
            ],
            stream=False
        )

        print(response.choices[0].message.content)




if __name__ == '__main__':
    # print("Hello, World!")
    callDeepSeek = CallDeepSeek()
    # callDeepSeek.test()
    callDeepSeek.test2()



















































