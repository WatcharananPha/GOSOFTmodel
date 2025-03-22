import streamlit as st
import os
import nest_asyncio
from llama_parse import LlamaParse
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings

nest_asyncio.apply()

st.set_page_config(
    page_title="ShopChat",
    page_icon="https://scontent.fbkk12-1.fna.fbcdn.net/v/t39.30808-6/450222371_913429780827473_1048735281141013489_n.jpg?_nc_cat=101&ccb=1-7&_nc_sid=6ee11a&_nc_ohc=kP6Im_OcBQAQ7kNvgEoiIW_&_nc_oc=AdmeLO7H-uUW9Q92rRm2nqt-wQHPd3U7_zjmvHICeV556w-sFz4RkEz8pvJK9g7IZpY&_nc_zt=23&_nc_ht=scontent.fbkk12-1.fna&_nc_gid=CqDJ3kyhGE7BMVW5-z-YVQ&oh=00_AYHlWMFasmuqgn5oZztWIVZZ5UWAMbVK8JMaRyoLlWkPeA&oe=67E418DA",
    layout="centered"
)

llm = ChatOpenAI(
    openai_api_key="sk-GqA4Uj6iZXaykbOzIlFGtmdJr6VqiX94NhhjPZaf81kylRzh",
    openai_api_base="https://api.opentyphoon.ai/v1",
    model_name="typhoon-v2-70b-instruct",
    temperature=1,
    max_tokens=8192,
)

parser = LlamaParse(
    api_key="llx-3QORP75OUx11inHUpIy67FLzIgYc0gjfAGKRLDiECXOXkkne",
    result_type="markdown",
    num_workers=1,
    verbose=True,
    language="en",
)

FAISS_INDEX_PATH = "faiss_index\index.faiss"

def process_uploaded_files(uploaded_files):
    all_text = ""
    
    for uploaded_file in uploaded_files:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                all_text += f.read() + "\n"
        else:
            documents = parser.load_data(file_path)
            for doc in documents:
                all_text += doc.text_resource.text + "\n"

        os.remove(file_path)
    
    return all_text

def get_vector_database(uploaded_files=None):
    embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    faiss_folder = "faiss_index"
    if os.path.exists(faiss_folder) and not uploaded_files:
        try:
            vector_store = FAISS.load_local(faiss_folder, embed_model, allow_dangerous_deserialization=True)
            return vector_store
        except Exception as e:
            st.error(f"Error loading FAISS index: {str(e)}")
            return None
    if uploaded_files:
        text_content = process_uploaded_files(uploaded_files)
        if not text_content:
            return None
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=256, 
            chunk_overlap=64
        )
        chunks = text_splitter.split_text(text_content)
        documents = [Document(page_content=chunk) for chunk in chunks]
        vector_store = FAISS.from_documents(
            documents=documents, 
            embedding=embed_model
        )
        os.makedirs(faiss_folder, exist_ok=True)
        vector_store.save_local(faiss_folder)
        
        return vector_store
        
    return None

def create_chatbot(vector_db):
    retriever = vector_db.as_retriever(search_kwargs={'k': 5}) if vector_db else None
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

    chat_prompt = ChatPromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": chat_prompt}
        )

    initial_response = qa_chain.invoke({
            "query": "งานในวันนี้มีอะไรบ้าง โปรดระบุรายการนัดหมายเข้าเยี่ยมลูกค้า การติดตามหนี้ และการจัดส่ง"
        })["result"]
    st.session_state.messages.append({
            "role": "assistant", 
            "content": f"👋 สวัสดีตอนเช้าค่ะ\n\n{initial_response}"
        })

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": chat_prompt}
    ) if retriever else None

def main():
    st.markdown(
        """
        <div style="text-align: center;">
            <img src="https://www.gosoft.co.th/wp-content/uploads/2019/01/cropped-LOGO-gosoft.png" width="500">
            <h1>ShopChat Sale Assistance</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    uploaded_files = st.file_uploader("Upload documents (optional)", accept_multiple_files=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    with st.spinner('Loading knowledge base...'):
        vector_db = get_vector_database(uploaded_files)
    if not vector_db and os.path.exists(FAISS_INDEX_PATH):
        st.warning("Failed to load the FAISS index. Please try re-uploading files.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    user_input = st.chat_input("Ask me anything...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
    
        if vector_db:
            chatbot = create_chatbot(vector_db)
            response = chatbot.run(user_input) if chatbot else "I'm unable to retrieve relevant data, but I'll do my best!"
        else:
            response = llm.predict(user_input)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()