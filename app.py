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
st.title("🤖 RetailGPT - Your Retail Guide..")

#persist_dir = 'chromadb_oadmin'
#folder='oadmin'
# load_dotenv()
from datetime import datetime

today = datetime.now()
print("Program Called Today's date:", today)
# persist_dir = 'chromadb_oconversion'
#st.write(os.listdir())

footer = """<style>
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
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#persist_dir=''
def prepare(folder, persist_dir):
    print("Preparing Embeddings for"+folder+ " in directory "+persist_dir+ " at " + str(datetime.now()))
    #st.write(folder, persist_dir)
    loader = PyPDFDirectoryLoader(folder)
    data = loader.load_and_split()
    #st.write(data)
    #st.write(os.listdir())
    #persist_dir= './chromadb_oconversion/'
    # print(persist_directory)
    context = "\n".join(str(p.page_content) for p in data)
    print("The total number of words in the context:", len(context))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
    context = "\n\n".join(str(p.page_content) for p in data)
    texts = text_splitter.split_text(context)
    # vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_dir)
    # vectordb = Chroma.from_documents(documents=pages, embedding=embeddings, persist_directory=persist_dir)
    # vectordb.persist()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_index = Chroma.from_texts(texts, embeddings, persist_directory=persist_dir)
    vector_index.persist()
    #vector_index = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    #print("Vector : "+str(vector_index.get())+" at " + str(datetime.now()))
    return vector_index
def load_default_context(folder, persist_dir):
     print(folder+persist_dir)
     st.session_state.vectordb=prepare(folder, persist_dir)
     print("All OK")

if 'context' not in st.session_state:
        #st.subheader("Setting up Default Context : Conversion")
        print("Default Context Setting Up"+str(datetime.now()))
        st.session_state.folder='oconversion'
        st.session_state.persist_dir='chromadb_oconversion'
        load_default_context(st.session_state.folder,st.session_state.persist_dir)
        st.session_state.context='Conversion'
        st.markdown(footer, unsafe_allow_html=True)
#load_sidebar()



def load_chroma(persist_dir, embeddings):
    with st.spinner(text="Loading indexed Retail Documents ! This should take 1-2 minutes."):
        #persist_dir = 'chromadb_oadmin'
        #st.write(persist_dir)
        vector_index = Chroma(persist_directory="./chromadb_oadmin", embedding_function=embeddings)
        #st.write(vector_index.get().keys())
    st.write(len(vector_index.get()["ids"]))
    # return vector_index

with st.sidebar:
    st.title("Settings")
    option = st.radio("Documentation",
                    ('Conversion', 'Administration'))
    print(option)
    print("New" + option)
    print("Old" + st.session_state.context)
    if st.session_state.context != option:
           print("New"+option)
           print("Old" + st.session_state.context)
           st.write(f"Selected Context: {option}")
           st.session_state.context = option
           if option == 'Conversion':
               with st.spinner("Processing"):
                   st.session_state.persist_dir = 'chromadb_oconversion'
                   st.session_state.folder = 'oconversion'
                   #st.session_state.context = "Conversion"
                   #vectordb = prepare(folder, persist_dir)
                   # vectordb=load_chroma(persist_directory1)
                   stt = "Current Context Set to :" + option + str(datetime.now())
                   print(stt)
                   st.session_state.vectordb = prepare(st.session_state.folder, st.session_state.persist_dir)
                   print("RJA" + stt)
                   print(st.session_state.vectordb.get().keys())
                   print(len(st.session_state.vectordb.get()["ids"]))
                   st.success("Indexing Completed!")
           elif option == 'User Guide':
                   st.session_state.persist_dir = './chromadb_oug'
                   # folder = 'oug'
                   #vectordb = prepare(folder, persist_dir)
           elif option == 'Administration':
                with st.spinner("Processing"):
                    st.session_state.persist_dir = 'chromadb_oadmin'
                    st.session_state.folder = 'oadmin'
                    #load_chroma(persist_dir, embeddings)
                    #st.session_state.context = "Administration"
                    # vectordb=load_chroma(persist_dir,embeddings)
                    stt = "Current Context Set to :"
                    print(stt)
                    st.session_state.vectordb = prepare(st.session_state.folder,st.session_state.persist_dir)
                    print("RJA" + stt)
                    print(st.session_state.vectordb.get().keys())
                    print(len(st.session_state.vectordb.get()["ids"]))
                    st.success("Indexing Completed!")
           elif option == 'Operations':
                st.session_state.persist_dir = 'chromadb_operations'
                # folder = 'operations'
           print(option)

print('embedding the document now')



with st.chat_message("user"):
    st.write("Hello User 👋 : Please wait while we initialize RetailGPT !")
print("Here!!!")

if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "RetailGPT is ready! Shoot your questions"}
    ]


def get_text():
    input_text = st.text_input("", key="input")
    return input_text


# st.write(persist_directory1)
# vectordb=load_chroma(persist_dir)
# vectordb=prepare(folder,persist_dir)
def search_chroma(question, persist_dir):
    print("Point 4.4.1.1")
    # result_docs = vectordb.similarity_search(query)
    #st.write("Raj" + persist_dir)
    print("Raj3" + persist_dir)
    print("Point 4.4.1.2")
    #st.write("Embed" + str(embeddings))
    #vectordb2 = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    #st.write(vectordb2.get().keys())
    #st.write(len(vectordb2.get()["ids"]))
    print("Raj" + str(question) + "Vijay")
    print("Point 4.4.1.3")
    #st.write("VECTOR" + str(vectordb2) + "Vijay")
    docs = st.session_state.vectordb.similarity_search(question)
    print("Point 4.4.1.4")
    prompt_template = """
      Answer the question as detailed as possible from the provided context, make sure to provide all the details\n\n
      Context:\n {context}?\n
      Question: \n{question}\n

      Answer:
    """
    print("Point 4.4.1.4")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    print("Point 4.4.1.5")
    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)
    print("Point 4.4.1.6")
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    # st.write("Raj2" + question + "Vijay2")
    #st.write(docs)
    print("Point 4.4.1.7")
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    # print(output)
    print("Point 4.4.1.8")
    print(response['output_text'])
    print("Point 4.4.1.9")
    return response['output_text']


# question = get_text()
# if question:
# output = search_chroma(vectordb,question)
# st.session_state.past.append(question)
# st.session_state.generated.append(output)

#
if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    print("Point 1")
    print("Questions Asked")
    st.session_state.messages.append({"role": "user", "content": prompt})
    print("Point 1.1")
for message in st.session_state.messages:  # Display the prior chat messages
    print("Point 2.1")
    with st.chat_message(message["role"]):
        print("Point 3.1")
        st.write(message["content"])
        print("Point 3.2")
    print("Point 2.2")
print("Point 1.3")
if st.session_state.messages[-1]["role"] != "assistant":
    print("Point 1.4")
    with st.chat_message("assistant"):
        print("Point 1.4.1")
        with st.spinner("Thinking..."):
            # response = chat_engine.chat(prompt)
            print(prompt+"and "+st.session_state.persist_dir)
            print("Point 1.4.1.1")
            output = search_chroma(prompt, st.session_state.persist_dir)
            print("Point 1.4.1.2")
            st.write(output)
            print("Point 1.4.1.3")
            message = {"role": "assistant", "content": output}
            print("Point 1.4.1.4")
            st.session_state.messages.append(message)  # Add response to message history
            print("Point 1.4.1.5")

#if __name__ == "__main__":
    #st.header("RetailGPT loading...")
    #print("Checking Context"+str(datetime.now()))
