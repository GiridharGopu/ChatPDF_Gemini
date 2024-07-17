import streamlit as st # Module for Webinterface for the chatapp
from PyPDF2 import PdfReader # This module helps in Reading and parsing the PDF Files
from langchain.text_splitter import RecursiveCharacterTextSplitter # This module helps in Splitting the data into chunks
import os # OS Module helps in communicating with the system files using for loading .env file variables
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Since we are using the Google Gemini Model we are using this module
import google.generativeai as genai # This helps in initialization the Model we are going to use 
from langchain_community.vectorstores import FAISS # This FAISS Vector Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI # Chat Model
from langchain.chains.question_answering import load_qa_chain # Chain to communicate 
from langchain.prompts import PromptTemplate # Prompt template to define a default Role to the model and to define the actions
from dotenv import load_dotenv # To load the Enviroment Variables

# Here loading the Enviroment Variables into the Module and configuring the Model
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Passing all our pdf Documents and Breaking them into chunks with page index to identify from which page we are extracting the Data chunks from.
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    
    if not text:
        st.warning("No text was extracted from the PDFs. They might be empty or protected.")
    else:
        st.info(f"Extracted {len(text)} characters from the PDFs.")
    return text


# 
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    if not chunks:
        st.error("No text chunks were created. The PDF might be empty or unreadable.")
        return None
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        st.error("No text chunks to process.")
        return
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Vector store created and saved successfully.")
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say,you can also provide your opinion on if the user askes on question, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with Documentation here")
    user_question = st.text_input("Currenty we fed the Model with data in PDF file")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text:
                    st.error("No text was extracted from the PDFs. Please check the files and try again.")
                else:
                    text_chunks = get_text_chunks(raw_text)
                    if text_chunks:
                        get_vector_store(text_chunks)
                        st.success("Processing completed successfully.")
                    else:
                        st.error("No text chunks were created. Processing failed.")

if __name__ == "__main__":
    main()