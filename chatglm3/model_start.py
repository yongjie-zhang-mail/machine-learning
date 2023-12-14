
from transformers import AutoTokenizer, AutoModel

class LLM1:

    def __init__(self):
        # C:\source\chatglm\ChatGLM3\model\chatglm3-6b
        self.model_path = "C:\source\chatglm\ChatGLM3\model\chatglm3-6b"
        # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        # model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True, device='cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(pretrained_model_name_or_path=self.model_path, trust_remote_code=True, device='cuda')
        self.model = self.model.eval()

    def model_chat_test1(self):
        response, history = self.model.chat(self.tokenizer, "你好", history=[])
        print(response)

        response, history = self.model.chat(self.tokenizer, "晚上睡不着应该怎么办", history=history)
        print(response)







if __name__ == '__main__':
    llm1 = LLM1()
    llm1.model_chat_test1()




