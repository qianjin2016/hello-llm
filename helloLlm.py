import time
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

api_base_url = "http://123.235.32.126:8889/v1" 
api_key = "EMPTY"
LLM_MODEL = "chatglm2-6b"
model = ChatOpenAI(
    streaming=True,
    verbose=True,
    # callbacks=[callback],
    openai_api_key=api_key,
    openai_api_base=api_base_url,
    model_name = LLM_MODEL,
    temperature=0.1
)

human_prompt = "{input}"
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [("human", "我们来玩成语接龙,上一个成语最后一个字作为下一个成语的开头一个字，我先来，生龙活虎"),
     ("ai", "虎头虎脑"),
     ("human", "{input}")])

start_time = time.time()
chain = LLMChain(prompt=chat_prompt, llm=model, verbose=True)
end_time = time.time()
print(chain({"input": "远近高低"}))
print("运行时间：", end_time - start_time, "秒")