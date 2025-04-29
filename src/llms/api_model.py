from src.configs.config import API_KEY
from openai import OpenAI

class API_LLM:
    def __init__(self):
        """
        初始化API语言模型
        """
        self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=API_KEY,
            )
        self.PREFIX = '''我需要适合RAG知识库的压缩文本，请你将文档内容进行去冗余、保留关键信息，只返回压缩后的文本。以下是文档内容：
                        '''

    @property
    def metadata(self):
        """返回模型元数据"""
        return {
            "model_name": "Qwen3 235B A22B",
            "context_window": 40960,  # Qwen3 235B A22B 的上下文窗口大小
            "num_output": 1280,  # 与 max_new_tokens 保持一致
        }

    def complete(self, prompt: str) -> str:
        """
        非流式生成文本
        
        Args:
            prompt: 输入提示词
            
        Returns:
            str: 生成的完整文本
        """
        try:
            content = self.PREFIX + prompt
            completion = self.client.chat.completions.create(
                model="qwen/qwen3-235b-a22b:free",
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"文本生成失败: {str(e)}")
