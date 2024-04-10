import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import openai
import json
from langchain.chains import LLMChain
with st.sidebar:
    st.title("PDF chat app")



def main():
    st.header("Chat with PDF")
    load_dotenv()
    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
            st.write("embedding loading from disk")
        else:
            embeddings = OpenAIEmbeddings()
            #model_name = "sentence-transformers/all-mpnet-base-v2"
            #embeddings = HuggingFaceEmbeddings(model_name=model_name)
            VectorStore = FAISS.from_texts(chunks,embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)
            st.write("embedding created and saved to disk")
    query = st.text_input("Ask a question about the PDF")
    if query:
        docs = VectorStore.similarity_search(query=query, k=3)
        st.write(docs)
        llm = OpenAI(temperature = 0,model = "gpt-3.5-turbo-instruct")
        chain = load_qa_chain(llm = llm)
        #chain = LLMChain(llm=llm)
        with get_openai_callback() as cb:
            response = chain.run(input_documents = docs, question = query)
            print(cb)
        st.write(response)

if __name__== '__main__':
    main()