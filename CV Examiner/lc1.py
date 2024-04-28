import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.callbacks import get_openai_callback



load_dotenv()


loader = CSVLoader(file_path="resumes_scores.csv")
documents = loader.load()
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)




def retrieve_score(query):
    similar_response = db.similarity_search(query, k=1)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array



llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
template = """
Give a score from 1 to 100 of the following resume:

{resume}

Here is a similar resume in the database with its score, based on that, give a score to the resume above:

{similar_resume}
"""
prompt = PromptTemplate(
    input_variables=["resume"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)



def generate_response(resume):
    similar_resume = retrieve_score(resume)
    response = chain.run(resume=resume, similar_resume = similar_resume)
    return response



def main():
    st.set_page_config(page_title="CV scoring")

    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    text = ""
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text()
            page_text_cleaned = ''.join(char for char in page_text if ord(char) < 128)
            text += page_text_cleaned.encode('utf-8', 'ignore').decode('utf-8')


        with get_openai_callback() as cb:
            response = generate_response(text)
            print(cb)
        st.write(response)

if __name__ == '__main__':
    main()