
import streamlit as st
from backend.rag import create_retriever, set_current_retriever
from utils.rag.index import extract_text_from_pdf, process_rag_query


def rag_page_component():
    st.title("📄 RAG - Document Q&A")
        
    # PDF Upload Section
    st.subheader("Upload PDF Document")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        if not st.session_state['pdf_uploaded'] or st.session_state['pdf_filename'] != uploaded_file.name:
            # New file uploaded or different file
            with st.spinner("Processing PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)
                
                if pdf_text:
                    # Create retriever from the extracted text
                    retriever = create_retriever(pdf_text)
                    
                    # Set the retriever globally (not in session state for serialization)
                    set_current_retriever(retriever)
                    
                    # Update session state
                    st.session_state['pdf_content'] = pdf_text
                    st.session_state['pdf_uploaded'] = True
                    st.session_state['pdf_filename'] = uploaded_file.name
                    st.success(f"✅ PDF '{uploaded_file.name}' processed successfully!")
                    
                    # Show preview of extracted text
                    with st.expander("Preview extracted text"):
                        st.text_area("Extracted content (first 1000 characters):", 
                                    value=pdf_text[:1000] + "..." if len(pdf_text) > 1000 else pdf_text,
                                    height=200, disabled=True)
                else:
                    st.error("Failed to extract text from PDF")

    # Display current PDF status
    if st.session_state['pdf_uploaded']:
        st.info(f"📄 Current document: {st.session_state['pdf_filename']}")
        
        # Clear PDF button
        if st.button("🗑️ Clear PDF", type="secondary"):
            st.session_state['pdf_uploaded'] = False
            st.session_state['pdf_content'] = ""
            st.session_state['pdf_filename'] = ""
            st.session_state['rag_history'] = []
            set_current_retriever(None)  # Clear global retriever
            st.rerun()

    # Chat interface for RAG
    if st.session_state['pdf_uploaded']:
        st.subheader("Ask questions about your document")
        
        # Display existing RAG messages
        for message in st.session_state['rag_history']:
            with st.chat_message(message['role']):
                st.markdown(message['content'])
        
        # RAG input
        rag_input = st.chat_input("Ask a question about the document...")
        
        if rag_input:
            # Add user message to RAG history
            st.session_state['rag_history'].append({'role': 'user', 'content': rag_input})
            with st.chat_message('user'):
                st.markdown(rag_input)
            
            # Process with RAG using LangGraph
            with st.spinner("Searching document and generating answer..."):
                rag_response = process_rag_query(rag_input)
            
            st.session_state['rag_history'].append({'role': 'ai', 'content': rag_response})
            with st.chat_message('ai'):
                st.markdown(rag_response)
    else:
        st.info("👆 Please upload a PDF document first to start asking questions.")