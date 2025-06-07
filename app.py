import os
import logging
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Store chat history and other state in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

def get_pdf_text(pdf_docs):
    """Extract and clean text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += (page.extract_text() or "").strip() + "\n"  # Clean and format text
    return text.strip()  # Final trim of any excess whitespace

def get_text_chunks(text):
    """Split text into chunks with predefined chunk size and overlap."""
    chunk_size = 10000
    chunk_overlap = 1000
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    """Create and save a vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    """Set up the question-answering chain with refined prompting."""
    prompt_template = """
    You are a knowledgeable assistant who provides clear and detailed answers based on the context provided. If the answer is not found in the context, politely inform the user and suggest they ask another question. Always strive for an informative and conversational tone.
    
    Context:\n{context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)  # Adjust temperature for creativity
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def summarize_pdf():
    """Generate a summary of the PDF content."""
    if st.session_state.vector_store is None:
        st.error("Vector store is not available. Please process PDFs first.")
        return
    
    try:
        new_db = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
                                   allow_dangerous_deserialization=True)
        docs = new_db.similarity_search("summarize the content of the PDF")

        context = "\n".join(f"Q: {entry['question']}\nA: {entry['answer']}" for entry in st.session_state.chat_history) + "\n"
        
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "context": context, "question": "Please summarize the content."}, return_only_outputs=True)

        st.session_state.chat_history.append({"question": "summarize the PDF", "answer": response["output_text"]})

        st.write("Summary: ", response["output_text"])
    except Exception as e:
        logger.error(f"Error in summarizing PDF: {str(e)}")
        st.error(f"An error occurred while summarizing the PDF: {str(e)}")

def answer_question(user_question):
    """Process the user question and return the answer."""
    if st.session_state.vector_store is None:
        st.error("Vector store is not available. Please process PDFs first.")
        return
    
    try:
        new_db = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
                                   allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        context = "\n".join(f"Q: {entry['question']}\nA: {entry['answer']}" for entry in st.session_state.chat_history) + "\n"
        
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)

        st.session_state.chat_history.append({"question": user_question, "answer": response["output_text"]})

        st.write("Reply: ", response["output_text"])
    except Exception as e:
        logger.error(f"Error in answering question: {str(e)}")
        st.error(f"An error occurred while answering the question: {str(e)}")

def display_chat_history():
    """Display the chat history."""
    if st.session_state.chat_history:
        st.write("### Chat History")
        for entry in st.session_state.chat_history:
            st.write(f"**Q:** {entry['question']}")
            st.write(f"**A:** {entry['answer']}")

def clear_chat_history():
    """Clear the chat history and vector store."""
    st.session_state.chat_history.clear()
    st.session_state.vector_store = None  # Clear the vector store
    st.success("Chat history cleared. You can now start a new conversation.")

def export_chat_history():
    """Export chat history to a CSV file."""
    if st.session_state.chat_history:
        df = pd.DataFrame(st.session_state.chat_history)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Chat History",
            data=csv,
            file_name='chat_history.csv',
            mime='text/csv',
        )
        st.success("Chat history ready for download.")
    else:
        st.warning("No chat history to export.")

def main():
    st.set_page_config("Chat PDF")
    st.header(" RAG  System🤖")

    # Input for user questions
    user_question = st.text_input("Ask a Question from the PDF Files")

    if st.button("Submit Question"):
        if user_question.lower() == "summarize the pdf":
            summarize_pdf()
        elif user_question:
            answer_question(user_question)
        else:
            st.warning("Please enter a question.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                    st.success("PDFs processed successfully.")
            else:
                st.error("Please upload at least one PDF file.")

        if st.button("Start New Conversation"):
            clear_chat_history()

        if st.button("Export Chat History"):
            export_chat_history()

        # Display chat history
        display_chat_history()

if __name__ == "__main__":
    main()
