
# https://python.langchain.com/v0.2/docs/integrations/chat/tongyi/

import os

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# pip install dashscope
import dashscope
from dashscope import TextEmbedding

# pip install --upgrade langchain-community "docarray"
# from langchain.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores import DocArrayInMemorySearch

# from langchain.embeddings import DashScopeEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings

from langchain.schema.runnable import RunnableMap





class AgentTest:

    def __init__(self):
        self.api_key = "sk-4d0ed669937c4e5bb13b5b55e41bc186"
        os.environ["DASHSCOPE_API_KEY"] = self.api_key
        dashscope.api_key=self.api_key


    def test_stream(self):
        chatLLM = ChatTongyi(            
            streaming=True,
        )
        # res = chatLLM.stream([HumanMessage(content="hi")], streaming=True)
        res = chatLLM.stream([HumanMessage(content="堺雅人")], streaming=True)
        for r in res:
            print("chat resp:", r)
    
    
    def simple_chain(self): 
        # 使用 ChatPromptTemplate 从模板创建一个提示，模板中的 {topic} 将在后续代码中替换为实际的话题
        prompt = ChatPromptTemplate.from_template(
            "告诉我一个关于{topic}的短笑话"
        )
        # 创建一个 ChatTongyi 模型实例
        model = ChatTongyi()
        # 创建一个StrOutputParser实例，用于解析输出
        output_parser = StrOutputParser()
        # 创建一个链式调用，将 prompt、model 和output_parser 连接在一起
        chain = prompt | model | output_parser
        # 调用链式调用，并传入参数
        message = chain.invoke({"topic": "熊"})
        print(message)
        print(chain)


    def generate_embeddings(self, text):
        rsp = TextEmbedding.call(model=TextEmbedding.Models.text_embedding_v1,
                                input=text)
        
        embeddings = [record['embedding'] for record in rsp.output['embeddings']]
        result = embeddings if isinstance(text, list) else embeddings[0]
        print(result)
        return result


    def complex_chain(self):         
        embeddings = DashScopeEmbeddings()
        db = DocArrayInMemorySearch.from_texts(
            ["哈里森在肯肖工作", "熊喜欢吃蜂蜜"],
            embedding=embeddings
        )
        query = "哈里森在哪里工作？"
        # query = "熊喜欢吃什么？"
        # result = db.similarity_search(query)
        
        retriever = db.as_retriever()
        result = retriever.invoke(query)

        print(result)


    def runnable_map_test(self, message):

        # 创建一个 ChatTongyi 模型实例
        model = ChatTongyi()
        # 创建一个StrOutputParser实例，用于解析输出
        output_parser = StrOutputParser()

        embeddings = DashScopeEmbeddings()
        db = DocArrayInMemorySearch.from_texts(
            ["哈里森在肯肖工作", "熊喜欢吃蜂蜜"],
            embedding=embeddings
        )                
        retriever = db.as_retriever()

        rm = RunnableMap({
            "context": lambda x: retriever.invoke(x["question"]),
            "question": lambda x: x["question"]
        })
        result = rm.invoke({"question": message})
        print(result)
        
        # 定义一个模板字符串template
        template = """仅根据以下上下文回答问题：
        {context}

        问题：{question}
        """
        # 使用 template 作为模板
        prompt = ChatPromptTemplate.from_template(template)

        chain = rm | prompt | model | output_parser

        result = chain.invoke({"question": message})
        # result = chain.invoke({"question": })
        print(result)      


    def bind_function(self): 
        # 定义一个函数
        functions = [
            {
                "name": "get_current_weather",
                "description": "获取指定位置的当前天气情况",
                "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                    "type": "string",
                    "description": "城市和省份，例如：北京，北京市",
                    },
                    "unit": {"type": "string", "enum": ["摄氏度", "华氏度"]},
                },
                "required": ["location"],
                },
            }
        ]






if __name__ == '__main__':
    # print("Hello, World!")
    agentTest = AgentTest()
    # agentTest.test_stream()
    # agentTest.simple_chain()
    # agentTest.generate_embeddings('AI PC 是指什么？')
    # agentTest.complex_chain()
    # 测试问题："哈里森在哪里工作？", "熊喜欢吃什么？"
    # message = "哈里森在哪里工作？"
    # agentTest.runnable_map_test(message=message)






