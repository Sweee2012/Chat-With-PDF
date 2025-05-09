import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings as HFEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

# Set page configuration
st.set_page_config(page_title="Chat with PDF 📄", page_icon="📄", layout="wide")


# Set your Google API key
api_key = "AIzaSyADbJYco4ivwQgFOFb_H6PjQDon9jmNC_M"
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
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
    embedding_model = HFEmbeddings(model_name='all-MiniLM-L6-v2')
    vector_store = FAISS.from_texts(text_chunks, embedding_model)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in the provided context, just say "Answer is not available in the context." Don't provide the wrong answer.
    Context: \n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.3,
        google_api_key=api_key
    )
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embedding_model = HFEmbeddings(model_name='all-MiniLM-L6-v2')
    new_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversation_chain()
    response = chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    st.write("🤖 **Reply:**", response["output_text"])

def main():
    st.header("📄 Chat with PDF using Gemini")
    user_question = st.text_input("❓ Ask a question from the PDF file")
    if user_question:
        user_input(user_question)
    with st.sidebar:
        st.title("📁 Menu:")
        pdf_docs = st.file_uploader("📤 Upload your PDF file(s) here and click on the 'Submit & Process' button", accept_multiple_files=True)
        if st.button("✅ Submit & Process"):
            with st.spinner("⏳ Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("✅ Done")

if __name__ == "__main__":
    main()
