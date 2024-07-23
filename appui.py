import streamlit as st
import streamlit_shadcn_ui as ui    
import os
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to process TXT files
def process_texts(files):
    texts = []
    for file in files:
        file_content = file.read().decode('utf-8')
        text = Document(page_content=file_content)
        texts.append(text)
    combined_text = "\n\n".join(text.page_content for text in texts)
    return combined_text, texts

# Function to process PDF files
def process_pdfs(files):
    texts = []
    for file in files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        texts.append(Document(page_content=text))
    combined_text = "\n\n".join(text.page_content for text in texts)
    return combined_text, texts

# Function to process CSV files
def process_csv(files):
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df, dfs

# Function to process Excel files
def process_excel(files):
    dfs = []
    for file in files:
        df = pd.read_excel(file)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df, dfs

# Function to process Word files
def process_word(files):
    texts = []
    for file in files:
        doc = docx.Document(file)
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        text = "\n".join(full_text)
        texts.append(Document(page_content=text))
    combined_text = "\n\n".join(text.page_content for text in texts)
    return combined_text, texts

# Function to chat with text files using LangChain and Google Generative AI
def chat_with_text_files(query, texts):
    if not texts:
        return "No texts available for processing."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150, add_start_index=True)
    docs_split = text_splitter.split_documents(texts)

    if not docs_split:
        return "Text splitting resulted in no documents."

    # Define embedding
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")    

    # Check if the embedding can be created successfully
    try:
        # Create vector database from data    
        db = FAISS.from_documents(docs_split, embedding=embedding)
    except IndexError:
        return "Error creating embeddings from the provided documents."

    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Initialize memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Define prompt template
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # QA CHAIN
    qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    
    # Run QA chain
    docs = db.similarity_search(query)
    if not docs:
        return "No similar documents found for the query."

    result = qa_chain({"input_documents": docs, "question": query})
    
    return result['output_text']

# Function to generate insights from a dataframe
def generate_insights(query, df):
    response = ""
    if query.lower() == "show dataframe":
        st.dataframe(df)
        response = "Displayed the dataframe."
    elif query.lower() == "show summary":
        st.write(df.describe())
        response = "Displayed the statistical summary of the dataframe."
    else:
        try:
            if "plot" in query.lower():
                col = query.split(" ")[-1]
                if col in df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(df[col], kde=True)
                    st.pyplot(plt)
                    response = f"Displayed the plot for column '{col}'."
                else:
                    response = f"Column '{col}' not found in the dataframe."
            else:
                response = "Query not recognized. Try 'show dataframe', 'show summary', or 'plot <column_name>'."
        except Exception as e:
            response = f"Error in processing the query: {e}"
    return response


# Page config
st.set_page_config(page_title="Chat with QueryBot", page_icon=":speech_balloon:", layout="wide")
# Toggle sidebar visibility
if st.button("⚙️ Settings"):
    st.session_state.show_settings = not st.session_state.show_settings
# Main Title
st.title("💬 Chat with QueryBot")
# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm your assistant named QueryBot. Click on the settings icon then connect to the database and start chatting."),
    ]
if 'show_settings' not in st.session_state:
    st.session_state.show_settings = False





# Sidebar settings
if st.session_state.show_settings:
    st.sidebar.subheader("Settings")
    st.sidebar.write("This is a chat application. Select a database type, connect, and start chatting.")

    # Dropdown for selecting database type
    db_type = st.sidebar.selectbox("Database Type", ["Select a database", "MySQL", "MongoDB", "CSV", "Excel", "PDF", "TXT"])

    # Database connection fields based on selected type
    if db_type == "MySQL":
        st.sidebar.text_input("Host", value="localhost", key="Host")
        st.sidebar.text_input("Port", value="3306", key="Port")
        st.sidebar.text_input("User", value="root", key="User")
        st.sidebar.text_input("Password", type="password", value="passcode", key="Password")
        st.sidebar.text_input("Database", value="RestaurantMenu", key="Database")
        if st.sidebar.button("Connect"):
            with st.spinner("Connecting to database..."):
                db = init_database(
                    st.session_state["User"],
                    st.session_state["Password"],
                    st.session_state["Host"],
                    st.session_state["Port"],
                    st.session_state["Database"]
                )
                st.session_state.db = db
                st.success("Connected to database!")

    elif db_type == "MongoDB":
        st.sidebar.text_input("Host", value="localhost", key="Mongo_Host")
        st.sidebar.text_input("Port", value="27017", key="Mongo_Port")
        st.sidebar.text_input("User", value="admin", key="Mongo_User")
        st.sidebar.text_input("Password", type="password", value="passcode", key="Mongo_Password")
        st.sidebar.text_input("Database", value="ClientData", key="Mongo_Database")
        if st.sidebar.button("Connect"):
            with st.spinner("Connecting to database..."):
                db = init_mongo_database(
                    st.session_state["Mongo_User"],
                    st.session_state["Mongo_Password"],
                    st.session_state["Mongo_Host"],
                    st.session_state["Mongo_Port"],
                    st.session_state["Mongo_Database"]
                )
                st.session_state.db = db
                st.success("Connected to database!")

    elif db_type == "CSV":
        csv_files = st.sidebar.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
        if csv_files:
            with st.spinner("Processing CSV files..."):
                combined_df, dfs = process_csv(csv_files)
                st.session_state.dfs = dfs
                st.session_state.df = combined_df
                st.success("CSV files processed!")

    elif db_type == "Excel":
        excel_files = st.sidebar.file_uploader("Upload Excel files", type=["xls", "xlsx"], accept_multiple_files=True)
        if excel_files:
            with st.spinner("Processing Excel files..."):
                combined_df, dfs = process_excel(excel_files)
                st.session_state.dfs = dfs
                st.session_state.df = combined_df
                st.success("Excel files processed!")

    elif db_type == "PDF":
        pdf_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        if pdf_files:
            with st.spinner("Processing PDF files..."):
                combined_text, texts = process_pdfs(pdf_files)
                st.session_state.texts = texts
                st.success("PDF files processed!")

    

    elif db_type == "TXT":
        text_files = st.sidebar.file_uploader("Upload TXT files", type=["txt"], accept_multiple_files=True)
        if text_files:
            with st.spinner("Processing text files..."):
                combined_text, texts = process_texts(text_files)
                st.session_state.texts = texts
                st.success("Text files processed!")

# Show chat history
for chat in st.session_state.chat_history:
    if isinstance(chat, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(f"✍️ {chat.content}")
    elif isinstance(chat, AIMessage):
        with st.chat_message("AI"):
            st.markdown(f"🟢 {chat.content}")

# Define a prompt template
template = """
Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in 
provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
Context:\n {context}?\n
Question: \n{question}\n

Answer:
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("user", "{text}"),
    ("assistant", "{response}")
])

# Function to handle specific words responses
specific_words_responses = {
    "hello querybot": "Hello! How can I assist you further?",
    "hi": "Hello! How can I help you today?",
    "hello": "Hello! How can I help you today?",
    "thank you": "You're welcome! If you have any other questions, feel free to ask.",
    "thanks": "You're welcome! If you have any other questions, feel free to ask.",
    "bye": "Goodbye! Have a great day!",
    "exit": "Goodbye! Have a great day!"
}

# Handle user input and interaction
if user_input := st.chat_input(placeholder="Ask your query..."):
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("Human"):
        st.markdown(f"✍️ {user_input}")

    response = specific_words_responses.get(user_input.lower())
    if response is None:
        with st.spinner("Processing..."):
            if db_type in ["TXT", "PDF", "Word"]:
                if "texts" in st.session_state:
                    texts = st.session_state.texts
                    response = chat_with_text_files(user_input, texts)
                else:
                    response = "No texts available for processing. Please upload text, PDF, or Word files."
            elif db_type in ["CSV", "Excel"]:
                if "df" in st.session_state:
                    df = st.session_state.df
                    response = generate_insights(user_input, df)
                else:
                    response = "No CSV or Excel file processed. Please upload CSV or Excel files."
            else:
                response = "No valid database or files loaded. Please check your settings."

    st.session_state.chat_history.append(AIMessage(content=response))
    with st.chat_message("AI"):
        st.markdown(f"🟢 {response}")
