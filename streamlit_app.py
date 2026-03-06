from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
import streamlit as st

client = ChatOpenAI(
    api_key=st.secrets['OPENAI_API_KEY'],
    model='deepseek-chat',
    base_url='https://api.deepseek.com',
    temperature=0.0
)

st.title("学习助手")
with st.sidebar:
    subject = st.selectbox("选择学科", options=['计算机', 'AI','数学与应用数学','抽象代数','高等代数','撩妹（恋爱）'])
    style = st.selectbox("讲解风格", options=['简洁', '详细'])
user_input = st.chat_input("请输入你的问题:")

# 初始化 Streamlit 会话消息历史
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = StreamlitChatMessageHistory(key="chat_history")
    st.session_state['chat_history'].add_ai_message("你好，我是你的学习助手!")

# 显示历史消息
for msg in st.session_state['chat_history'].messages:
    st.chat_message(msg.type).write(msg.content)

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
    # 新版提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    return prompt.partial(subject=subject, style=style_dict[style])

def generate_response(user_input, subject, style):
    # 1. 获取提示词模板
    prompt = get_prompt_template(subject, style)
    # 2. 构建基础链
    chain = prompt | client
    # 3. 包装成带历史的对话链（核心替代 ConversationChain）
    chat_chain = RunnableWithMessageHistory(
        chain,
        lambda session_id: st.session_state['chat_history'],
        input_messages_key="input",
        history_messages_key="history"
    )
    # 4. 调用链
    response = chat_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "default"}}
    )
    return response.content

# 处理用户输入
if user_input:
    st.chat_message('human').write(user_input)
    with st.spinner('AI正在思考中，请稍候......'):
        response = generate_response(user_input, subject, style)
    st.chat_message('ai').write(response)

