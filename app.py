
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceBgeEmbeddings
import streamlit as st
from langchain.document_loaders import PyPDFLoader
import pdfplumber
import tempfile

model_path="google/flan-t5-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def file_processing(file):

    # Load data from PDF
    loader=PyPDFLoader(file)
    data=loader.load_and_split()

    content = ''
    for page in data:
        content += page.page_content

    splitter_ans = TokenTextSplitter(
        chunk_size = 250,
        chunk_overlap = 20
    )
    chunks_ans = splitter_ans.split_text(content)
    document_ans = [Document(page_content=t) for t in chunks_ans]

    ans_gen = splitter_ans.split_documents(
        document_ans
    )    
    
    
    return ans_gen

def llm_pipeline(file_path):
    
    Ans_doc = file_processing(file_path)

    llm = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=128
    )

    llm_model = HuggingFacePipeline(pipeline=llm)

    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_store = FAISS.from_documents(Ans_doc, embeddings)

    ans_chain = RetrievalQA.from_chain_type(llm=llm_model, 
                                                chain_type="stuff", 
                                                retriever=vector_store.as_retriever())

    return ans_chain 


## streamlit code
def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


st.set_page_config(layout='wide',page_title='QA-web-APP')
def main():
    st.title('PDF Question Answer Web-App')

    uploaded_file=st.file_uploader("Upload your PDF File Here",type=['pdf'])

    if uploaded_file is not None:

        question = st.text_input("Enter your question:")
        if st.button("Show Q and A"):
            filepath =uploaded_file.name
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                filepath = tmp_file.name
            ans_chain = llm_pipeline(filepath)
            answer = ans_chain.run(question)
            st.text("____Answer____")
            st.text(answer)

if __name__ == '__main__':
            main()
