# 新版 LangChain 正确导入路径
from langchain_community.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
import streamlit as st

# 初始化 DeepSeek 客户端（新版写法）
client = ChatOpenAI(
    api_key=st.secrets['OPENAI_API_KEY'],
    model='deepseek-chat',
    base_url='https://api.deepseek.com',
    temperature=0.0
)

st.title("学习助手")
with st.sidebar:
    subject = st.selectbox("选择学科", options=['计算机', 'AI'])
    style = st.selectbox("讲解风格", options=['简洁', '详细'])
user_input = st.chat_input("请输入你的问题:")

# 初始化会话状态（兼容新版 Memory）
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role':'ai','content':'你好，我是你的学习助手!'}]
    # 新版 ConversationSummaryMemory 初始化（参数不变）
    st.session_state['memory'] = ConversationSummaryMemory(
        memory_key='chat_history',
        return_messages=True,
        llm=client  # 新版需要显式传入 LLM 用于总结
    )

# 显示历史消息
for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])

def get_prompt_template(subject, style):
    style_dict = {
        "简洁": "仅提供直接答案和最少的必要解释。不要添加额外细节、发散讨论或无关信息。保持回答清晰、简洁，目标是为用户快速提供解决方案。",
        "详细": "第一，针对用户提问给出直接答案和清晰的解释；第二，基于此提供必要的相关知识点的信息，以补充背景或加深理解。",
    }
    system_template = """你是{subject}领域的专家，只回答与{subject}学科相关的问题。

【重要规则】
1. 如果用户提问与{subject}学科无关，你必须直接拒绝回答，不要提供任何相关信息
2. 拒绝时使用固定格式："抱歉，我只负责{subject}学科相关的问题，您的问题属于其他领域，我无法解答。"
3. 不要尝试回答无关问题的任何部分
4. 如果问题边界模糊，优先判断是否与{subject}核心知识相关

【学科范围】
- 计算机/AI 学科包括：编程、算法、数据结构、人工智能、机器学习、深度学习、神经网络等
- 其他学科如：语文、数学、历史、地理、物理、化学、生物等均不属于本学科范围

【讲解风格】
{style}

请严格遵守以上规则。"""
    # 新版 ChatPromptTemplate 写法（兼容旧版）
    prompt_template = ChatPromptTemplate.from_messages([
        ('system', system_template),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{input}')
    ])
    # 绑定部分变量（新版写法）
    prompt_template = prompt_template.partial(subject=subject, style=style_dict[style])
    return prompt_template

def generate_response(user_input, subject, style, memory):
    prompt = get_prompt_template(subject, style)
    # 新版 ConversationChain 初始化（参数不变）
    chain = ConversationChain(llm=client, memory=memory, prompt=prompt)
    # 新版 invoke 调用（返回格式不变）
    response = chain.invoke({'input': user_input})
    return response['response']

# 处理用户输入
if user_input:
    st.chat_message('human').write(user_input)
    st.session_state['messages'].append({'role':'human','content':user_input})

    with st.spinner('AI正在思考中，请稍候......'):
        response = generate_response(user_input, subject, style, st.session_state['memory'])
    st.chat_message('ai').write(response)
    st.session_state['messages'].append({'role':'ai','content':response})
