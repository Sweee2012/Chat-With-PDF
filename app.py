import streamlit as st
from PyPDF2 import PdfReader
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Set page configuration
st.set_page_config(page_title="Chat with PDF 📄", page_icon="📄", layout="wide")


# Set your Google API key
api_key = "AIzaSyADbJYco4ivwQgFOFb_H6PjQDon9jmNC_M"  # Replace with your actual API key

def get_pdf_text(uploaded_file):
    text = ""
    reader = PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=4,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vector_store = FAISS.from_texts(text_chunks, embedding_model)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    prompt_template = """
    Answer the following question based on the provided context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_messages([("system", prompt_template)])
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain

def main():
    st.title("Chat with Your PDF 📄")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file is not None:
        text = get_pdf_text(uploaded_file)
        text_chunks = get_text_chunks(text)
        get_vector_store(text_chunks)

        st.success("PDF processed successfully! You can now ask questions.")

        user_question = st.text_input("Ask a question about the PDF:")
        if user_question:
            chain = get_conversation_chain()
            response = chain.invoke({"context": text, "question": user_question})
            st.write("Answer:", response)

if __name__ == "__main__":
    main()
