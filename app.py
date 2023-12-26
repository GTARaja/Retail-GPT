import streamlit as st
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from streamlit_chat import message
from dotenv import load_dotenv
import os
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import UnstructuredPDFLoader
persist_dir='chromadb_oadmin'
#folder='oadmin'
#load_dotenv()
model = genai.GenerativeModel('gemini-pro')
#persist_dir = 'chromadb_oconversion'
st.write(os.listdir())
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
def prepare(folder,persist_dir):
    st.write(folder,persist_dir)
    loader = PyPDFDirectoryLoader(folder)
    data = loader.load_and_split()
    st.write(data)
    st.write(os.listdir())
    #persist_directory = './chromadb_oconversion'
    #print(persist_directory)
    context = "\n".join(str(p.page_content) for p in data)
    print("The total number of words in the context:", len(context))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    context = "\n\n".join(str(p.page_content) for p in data)
    texts = text_splitter.split_text(context)
    # vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_dir)
    #vectordb = Chroma.from_documents(documents=pages, embedding=embeddings, persist_directory=persist_dir)
    # vectordb.persist()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = Chroma.from_texts(texts, embeddings,persist_directory=persist_dir)
    vector_index.persist()
    vector_index = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vector_index

def load_chroma(persist_dir,embeddings):
    with st.spinner(text="Loading indexed Retail Documents ! This should take 1-2 minutes."):
        persist_dir = 'chromadb_oadmin'
        st.write(persist_dir)
        vector_index = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return vector_index
    
with st.sidebar:
    st.title("Settings")
    #ques = st.radio( "Documentation",
    #('Conversion','User Guide','Administration','Operations'))
    ques = st.radio( "Documentation",
    ('Conversion','Administration'))
    if st.button("Process"):
        with st.spinner("Processing"):
            if ques == 'Conversion':
                persist_dir = './chromadb_oconversion'
                folder = 'oconversion'
                vectordb=prepare(folder,persist_dir)
                #vectordb=load_chroma(persist_directory1)
            if ques == 'User Guide':
                persist_dir = './chromadb_oug'   
                #folder = 'oug'
                vectordb=prepare(folder,persist_dir) 
            if ques == 'Administration':
                persist_dir = './chromadb_oadmin'  
                folder = 'oadmin'
                vectordb=load_chroma(persist_dir,embeddings)
                #vectordb=prepare(folder,persist_dir)
            if ques == 'Operations':
                persist_dir = 'chromadb_operations'  
                #folder = 'operations'
            st.success("Done")


for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)




print('embedding the document now')

st.title("🤖 RetailGPT - Your Retail Guide..")

with st.chat_message("user"):
    st.write("Hello User 👋 : Please wait while we initialize RetailGPT !")
print("Here!!!")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "RetailGPT is ready! Shoot your questions"}
    ]

def get_text():
    input_text = st.text_input("", key="input")
    return input_text



#st.write(persist_directory1)
#vectordb=load_chroma(persist_dir)
#vectordb=prepare(folder,persist_dir)
def search_chroma(question,persist_dir):
    #result_docs = vectordb.similarity_search(query)
    st.write("Raj"+persist_dir)
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    print("Raj"+str(question)+"Vijay")

    docs = vectordb.similarity_search(question)
    prompt_template = """
      Answer the question as detailed as possible from the provided context, make sure to provide all the details\n\n
      Context:\n {context}?\n
      Question: \n{question}\n

      Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    #st.write("Raj2" + question + "Vijay2")
    #st.write(docs)
    response = chain({"input_documents": docs, "question": question},return_only_outputs=True)
    #print(output)
    print(response['output_text'])
    return response['output_text']

#question = get_text()
#if question:
    #output = search_chroma(vectordb,question)
    #st.session_state.past.append(question)
    #st.session_state.generated.append(output)

#
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            #response = chat_engine.chat(prompt)
            output = search_chroma(prompt,persist_dir)
            st.write(output)
            message = {"role": "assistant", "content": output}
            st.session_state.messages.append(message) # Add response to message history




footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;'target="_blank">Raj "GTARaja" Vijay</a></p>

</div>
"""
st.markdown(footer,unsafe_allow_html=True)
