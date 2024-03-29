# from langchain_openai import OpenAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.chains import ConversationChain
# from langchain.chains.router.llm_router import LLMRouterChain,RouterOutputParser
# from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
# from langchain.chains.router import MultiPromptChain


# coder_template = """你是一个非常专业的程序员，对于各种代码实现需求，你会编写精致的代码.
# 你编写的代码，每一行都有相对应的注释，让人一目了然。
# 而且你的回复里只有代码和注释，不会说其他的话。
# 接下来是代码的需求:
# {input}"""

# math_template = """你是一个数学家，对于各种数学问题，你都能够解决。
# 接下来是数学问题:
# {input}"""


# info=[
#     {
#         "name":"coder",
#         "template":coder_template,
#         "description":"专业的程序员"
#     },
#     {
#         "name":"math",
#         "template":math_template,
#         "description":"数学家"
#     }
# ]

# llm=OpenAI(model="gpt-3.5-turbo-instruct",temperature=0.5)

# destination_chains ={}
# for i in info:
#     name=i["name"]
#     template=PromptTemplate(template=i["template"],input_variables=["input"])
#     chain = LLMChain(llm=llm,prompt=template)
#     destination_chains[name]=chain

# default_chain=ConversationChain(llm=llm,output_key="text")


# destinations=[f"{i['name']}:{i['description']}" for i in info]
# destinations_str="\n".join(destinations)
# router_prompt=MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
# router_prompt=PromptTemplate(
#     template=router_prompt,
#     input_variables=["input"],
#     output_parser=RouterOutputParser()
# )
# router_chain=LLMRouterChain.from_llm(llm,router_prompt)

# chain=MultiPromptChain(
#     router_chain=router_chain,
#     destination_chains=destination_chains,
#     default_chain=default_chain,
#     verbose=True
# )





# import gradio as gr
# import time

# def echo(message, history):
#     result=chain.run(message)
#     for i in range(len(result)):
#         time.sleep(0.02)
#         yield result[: i+1]

#         #自定义的流式输出



# demo = gr.ChatInterface(fn=echo, 
#                         examples=["hello", "hola", "merhaba","使用python写快速排序算法"], 
#                         title="Echo Bot",
#                         theme="soft")
# demo.launch()






from langchain_openai import OpenAI,ChatOpenAI
from langchain.prompts import PromptTemplate 
from langchain.chains import SequentialChain,LLMChain

llm=ChatOpenAI(model="gpt-3.5-turbo-0125",temperature=1)
template1="""你是一个玄幻小说家,对于一个小说标题,你会首先给出十个章节标题,后续再进行深度编写
对于玄幻小说标题为<{input}>的小说,请给出十个章节标题.注意章节标题不超过15个字.
格式为:1.第一章:xxx 2.第二章:xxx...."""

template2="""你是一个玄幻小说家,对于不同的小说章节标题,你会根据每一个标题,注意每一个标题都编写出1500字左右的具体内容!!!
注意每一个标题都编写出1500字左右的具体内容!!!
而且每一个相邻章节之间,内容必须要有关系,不能断裂.
小说的大标题是<{input}>,请根据标题:{topic},给出每一章的内容.
格式为:第一章内容:xxx  第一章内容:xxx.... """

template1=PromptTemplate(template=template1,input_variables=["input"])
chain_write_theme=LLMChain(llm=llm,prompt=template1,output_key="topic",verbose=True)


template2=PromptTemplate(template=template2,input_variables=["input","topic"])
chain_write_content=LLMChain(llm=llm,prompt=template2,output_key="content",verbose=True)

chain=SequentialChain(chains=[chain_write_theme,chain_write_content],input_variables=["input"],output_variables=["content"],verbose=True)



import gradio as gr
import time

def echo(message, history):
    result=chain.run({"input":message})
    for i in range(len(result)):
        time.sleep(0.01)
        yield result[: i+1]

        #自定义的流式输出



demo = gr.ChatInterface(fn=echo, 
                        examples=["我觉醒了修仙系统", "我在都市当兵王","我在异世界修仙"], 
                        title="我是超级玄幻小说家",
                        theme="soft")
demo.launch()
