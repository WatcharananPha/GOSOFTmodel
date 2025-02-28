import streamlit as st
import warnings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "shopchat"
if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(
    page_title="ShopChat Admin",
    page_icon="https://www.gosoft.co.th/wp-content/uploads/2019/01/cropped-LOGO-gosoft.png",
    layout="centered"
)

model_name = "BAAI/bge-m3"
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

customer_db = FAISS.load_local("customer_index", embeddings, allow_dangerous_deserialization=True)
crm_db = FAISS.load_local("crm_index", embeddings, allow_dangerous_deserialization=True)
retriever = customer_db.as_retriever(search_kwargs={"k": 2})

llm = ChatOpenAI(
    api_key='sk-GqA4Uj6iZXaykbOzIlFGtmdJr6VqiX94NhhjPZaf81kylRzh',
    base_url='https://api.opentyphoon.ai/v1',
    model_name="typhoon-v2-70b-instruct"
)

template = """
You are a Salesman Assistant AI designed to enhance a CRM system.
Your responsibilities include:
- Reminding salespeople of daily tasks
- Providing customer and product details
- Planning customer visits
- Managing orders and tracking deliveries
- Following up on debts and payments
- Surveying the market and tracking competitor prices
- Assisting with customer communication
- Answering questions about products

Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use the retrieved information and avoid making up answers.

Retrieved context: {context}

Salesperson Query: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

st.markdown("""
    <style>
        .chat-container { max-width: 500px; margin: auto; }
        .user-bubble {
            background-color: #de152c;
            padding: 10px;
            border-radius: 15px;
            max-width: 80%;
            margin-bottom: 10px;
            float: right;
            clear: both;
            color: white;
        }
        .assistant-bubble {
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 15px;
            max-width: 80%;
            margin-bottom: 10px;
            float: left;
            clear: both;
        }
        .bottom-nav {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: black;
            display: flex;
            justify-content: space-around;
            padding: 10px 0;
            border-radius: 20px 20px 0 0;
        }
        .nav-item {
            color: gray;
            font-size: 12px;
            text-align: center;
            cursor: pointer;
        }
        .nav-item img {
            width: 24px;
            height: 24px;
        }
        .selected { color: red; }
        .title-container {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-container">ShopChat Admin</div>', unsafe_allow_html=True)
for msg in st.session_state.messages:
    bubble_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f'<div class="{bubble_class}">{msg["content"]}</div>', unsafe_allow_html=True)

user_input = st.chat_input("Ask about customers or CRM contacts...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f'<div class="user-bubble">{user_input}</div>', unsafe_allow_html=True)
    with st.spinner("Fetching customer insights..."):
        try:
            response = qa_chain.invoke({"query": user_input})["result"]
        except Exception as e:
            response = f"Error: {e}"
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.markdown(f'<div class="assistant-bubble">{response}</div>', unsafe_allow_html=True)

def set_tab(tab_name):
    st.session_state.selected_tab = tab_name

tab_titles = {
    "3d": "3D Service",
    "schedule": "Schedule",
    "chatbot": "Chatbot",
    "more": "อื่นๆ"
}

if st.session_state.selected_tab in tab_titles:
    st.title(tab_titles[st.session_state.selected_tab])

shopchat = "selected" if st.session_state.selected_tab == "shopchat" else ""
three_d = "selected" if st.session_state.selected_tab == "3d" else ""
schedule = "selected" if st.session_state.selected_tab == "schedule" else ""
chatbot = "selected" if st.session_state.selected_tab == "chatbot" else ""
more = "selected" if st.session_state.selected_tab == "more" else ""

st.markdown(f"""
    <div class="bottom-nav">
        <div class="nav-item {shopchat}" onclick="setTab('shopchat')">
            <img src="https://cdn-icons-png.flaticon.com/512/25/25694.png" alt="ShopChat"><br>
            ShopChat
        </div>
        <div class="nav-item {three_d}" onclick="setTab('3d')">
            <img src="https://cdn-icons-png.flaticon.com/512/103/103093.png" alt="3D"><br>
            3D
        </div>
        <div class="nav-item {schedule}" onclick="setTab('schedule')">
            <img src="https://cdn-icons-png.flaticon.com/512/2991/2991112.png" alt="Schedule"><br>
            Schedule
        </div>
        <div class="nav-item {chatbot}" onclick="setTab('chatbot')">
            <img src="https://cdn-icons-png.flaticon.com/512/2950/2950745.png" alt="Chatbot"><br>
            Chatbot
        </div>
        <div class="nav-item {more}" onclick="setTab('more')">
            <img src="https://cdn-icons-png.flaticon.com/512/566/566048.png" alt="More"><br>
            อื่นๆ
        </div>
    </div>
    
    <script>
        function setTab(tabName) {{
            const params = new URLSearchParams(window.location.search);
            params.set('_session', JSON.stringify({{"selected_tab": tabName}}));
            fetch('?' + params.toString(), {{method: 'POST'}}).then(() => {{
                window.location.reload();
            }});
        }}
    </script>
""", unsafe_allow_html=True)

if "initialized" not in st.session_state:
    st.session_state.initialized = False

st.markdown("""
    <style>
        /* Existing CSS styles... */
        
        .suggestion-button {
            background-color: #f0f0f0;
            border: none;
            padding: 8px 16px;
            margin: 4px;
            border-radius: 20px;
            cursor: pointer;
            font-size: 14px;
        }
        
        .suggestion-button:hover {
            background-color: #de152c;
            color: white;
        }
        
        .suggestions-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 16px 0;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

if not st.session_state.initialized:
    try:
        initial_response = qa_chain.invoke({
            "query": "What are the tasks today? Please list all scheduled customer visits, payments due, and deliveries."
        })["result"]
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"👋 Good morning! Here are your tasks for today:\n\n{initial_response}"
        })
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Good morning! I'm ready to help you with your tasks today."
        })
    st.session_state.initialized = True

st.markdown('<div class="suggestions-container">', unsafe_allow_html=True)
suggestions = [
    "ข้อมูลลูกค้าที่ต้องเข้าพบ", 
    "ความสนใจของลูกค้า"
]

for suggestion in suggestions:
    if st.button(suggestion, key=suggestion, help=f"Click to ask about {suggestion}"):
        st.session_state.messages.append({"role": "user", "content": suggestion})
        
        with st.spinner("Getting information..."):
            try:
                response = qa_chain.invoke({"query": suggestion})["result"]
            except Exception as e:
                response = f"Error: {e}"
                
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)