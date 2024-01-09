import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

#load_dotenv()
#os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
st.title("🤖 RetailGPT - Your Retail Guide..")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
new_db = FAISS.load_local("faiss_index", embeddings)
with st.chat_message("RetailGPT"):
    st.write("Hello User 👋 : Please wait while we initialize RetailGPT !")
print("Here!!!")

if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = [
        {"role": "RetailGPT", "content": "RetailGPT is ready! Shoot your questions"}
    ]
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    try to find some reference and answer accurately.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=1)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):



    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    #st.write("Reply: ", response["output_text"])
    return response["output_text"]




if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    print("Point 1")
    print("Questions Asked")
    st.session_state.messages.append({"role": "User", "content": prompt})
    print("Point 1.1")
for message in st.session_state.messages:  # Display the prior chat messages
    print("Point 2.1")
    with st.chat_message(message["role"]):
        print("Point 3.1")
        st.write(message["content"])
        print("Point 3.2")
    print("Point 2.2")
print("Point 1.3")
if st.session_state.messages[-1]["role"] != "RetailGPT":
    print("Point 1.4")
    with st.chat_message("RetailGPT"):
        print("Point 1.4.1")
        with st.spinner("Thinking..."):
            # response = chat_engine.chat(prompt)
            #print(prompt+"and "+st.session_state.persist_dir)
            print("Point 1.4.1.1")
            output = user_input(prompt)
            print("Point 1.4.1.2")
            st.write(output)
            print("Point 1.4.1.3")
            message = {"role": "RetailGPT", "content": output}
            print("Point 1.4.1.4")
            st.session_state.messages.append(message)  # Add response to message history
            print("Point 1.4.1.5")

