from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback

def main():
    # Load environment variables
    load_dotenv()

    # Set Streamlit page configuration
    st.set_page_config(page_title="LangChain PDF Explorer")
    st.header("LangChain PDF Explorer")
    
    # Upload PDF file
    pdf = st.file_uploader("Upload PDF", type="pdf")
    
    if pdf is not None:
      # Read the PDF file
      pdf_reader = PdfReader(pdf)
      text = ""
      # Extract text from each page of the PDF
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # Split the text into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # Generate embeddings for the chunks
      embeddings = OpenAIEmbeddings()
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # Get user's question
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        # Search for similar documents in the knowledge base
        docs = knowledge_base.similarity_search(user_question)
        
        # Load the question answering chain
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        input_data = {"input_documents": docs, "question": user_question}

        # Invoke the chain and get the response
        with get_openai_callback() as cb:
          response = chain.invoke(input=input_data)
          print(cb)
          cb.total_cost = cb.total_cost.__round__(7)
          st.write(cb)

        # Display the response
        st.write(response['output_text'])

# Run the main function
if __name__ == '__main__':
    main()