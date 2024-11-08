import streamlit as st
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from htmltemplate import css,bot_template,user_template
load_dotenv()

# groq_api_key = st.secrets["groq_api_key"]
groq_api_key = os.getenv('GROQ_API_KEY')
# Initialize the language model

def get_pdf_text(pdf_docs):
    text= ""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        # pdf_reader contains the pages from pdf's
        for page in pdf_reader.pages:
            # extract_text()- extracts text from pdf
            text+=page.extract_text()
    
    return text

def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len)
    chunks=text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings=HuggingFaceBgeEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore=FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore
    
def get_conversation_chain(vectorstore):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    memory=ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory)
    return conversation_chain

def handle_userinput(user_question):
    response=st.session_state.conversation({'question':user_question})
    st.session_state.chat_history=response["chat_history"]
    
    for i, message in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat With Multiple Pdf's",
                    page_icon=":books:",layout = "centered")
    st.write(css,unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation=None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat With Multiple PDF's :books:")
    user_question=st.text_input("Ask a question about the your document here",placeholder="Type here")
    
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs=st.file_uploader("Upload your PDF's here and press click on 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get the data
                raw_text=get_pdf_text(pdf_docs)
                # st.write(raw_text)
                
                # get the text chunks
                text_chunks=get_text_chunks(raw_text)
                # st.write(text_chunks)
                
                # create vector store
                vectorstore=get_vectorstore(text_chunks)
                st.session_state.conversation=get_conversation_chain(vectorstore)
        
if __name__=="__main__":
    main()