# Agent开发

## 背景

可以将一个Agent看做是一个简单的系统：

1. 对用户的输入输出响应
2. 在系统内部，根据用户的输出可以做以下事情：
   1. RAG
   2. 工具调用
   3. 读写记忆

## MVP版本

这个版本就是处理用户的输入，然后生成文本

```python

class Message:
    pass

class Session:

    def add_message(msg: Message):
        pass

    def to_llm_prompt() -> List[Dict[str, Any]]:
        pass

class Agent:

    def __init__(
        self,
        model: str,
        session_id: str = None
    ):
        self.session = Session()

    def chat(user_input: str) -> str:
        pass

```

一个简单的Agent:

1. 可以响应用户的问题
2. 可以对用户的问题进行思考
3. 如果可以直接回答问题，就回答，如果不能，看能否有可以的工具调用
