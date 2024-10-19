import streamlit as st
import openai
import os
from langchain.llms import OpenAI,GooglePalm
import langchain
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain,ConversationalRetrievalChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import TextLoader,CSVLoader,UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
import pickle


OPENAI_API_KEY=''
llm = OpenAI(openai_api_key=OPENAI_API_KEY,temperature=0.3,max_tokens=500)
# llm =GooglePalm(google_api_key=api_key,temperature=0.2)

file_path='fiass_store_openai.pkl'


st.title("Url Scraping Tool 📈")

st.sidebar.title("Enter your Urls")




add_button=st.sidebar.button(label="➕ Add a url")

urls=[st.sidebar.text_input(f"URL 1",key=1)]


if add_button:
    len_of_urls=len(urls)
    url=st.sidebar.text_input(f"URL {len_of_urls+1}",key=len_of_urls+1)
    urls.append(url)
    len_of_urls+=1

len_of_urls=1


process_url_clicked=st.sidebar.button("Process Urls")

main_placeholder=st.empty()

if process_url_clicked:
    if not urls[0]:
        st.sidebar.write("Please first paste some source urls 😒")
        st.stop()
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text('Data loading started....✅')
    data=loader.load()

    # splitting the data
    text_splitters=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',","],
        chunk_size=1000,
        chunk_overlap=200
    )

    docs=text_splitters.split_documents(data)
    
    #create embeddings and saving into Faiss index
    # embeadding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    embeadding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
    vectorstore_openai__=FAISS.from_documents(docs,embeadding)

    main_placeholder.text('Embeadding Vector Started Building ....✅✅')

    # saving in the pickle format
    
    with open(file_path,'wb') as f:
        pickle.dump(vectorstore_openai__,f)


query=main_placeholder.text_input('Question: ')
if query:
    if os.path.exists(file_path):
        with open(file_path,'rb') as f:
            vectorStore=pickle.load(f)

            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorStore.as_retriever(),verbose=True)
            result=chain({'question':query},return_only_outputs=True)

            st.text("Answer:")
            st.write(result['answer'])

            sources=result.get('sources',None)
            if sources:
                st.text('Sources:')
                sources_list=sources.split('\n')
            for source in sources_list:
                st.write(f"[{result['sources'][:50]}.....]({result['sources']})")


    
    # load the vector database