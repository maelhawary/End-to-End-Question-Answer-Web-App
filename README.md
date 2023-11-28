# Question-Answer-Web-App

This is an LLM project for answering questions given any document context. By running the App, you can upload any Pdf document then getting a reply for any question that is related to the article. The LLM model is based on google-flan-t5-base (https://huggingface.co/google/flan-t5-base). The implementation of this project includes the following packages: Langchain and Huggingface for data preprocessing and model integeration, and Streamlit for web-app interface. 

# To use
Install requirements using,
```bash
  pip install -r requirements.txt
``
Then, run the app as follows:
```bash
    streamlit run web-app.py
```
