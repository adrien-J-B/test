import streamlit as st
import os
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import tempfile

# Configure the page
st.set_page_config(page_title="Mental Health Q&A", page_icon="❓")

st.title("Mental Health Resources Q&A ❓")
st.write("Ask questions about student mental health and get answers from verified resources.")

# Initialize Gemini API (you'll need to set up an API key)
@st.cache_resource
def setup_gemini():
    try:
        # You'll need to set up your Gemini API key
        # genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        # For now, we'll use a mock response if no API key is available
        model = genai.GenerativeModel('gemini-pro')
        return model
    except:
        st.warning("Gemini API not configured. Using mock responses for demonstration.")
        return None

# Load and process documents
@st.cache_resource
def load_documents():
    # Sample mental health resources (in a real app, you'd load from files)
    resources = {
        "WHO Student Mental Health Guide": """
        Student Mental Health: A Guide for Universities
        
        Mental health is a state of well-being in which an individual realizes their own abilities, 
        can cope with the normal stresses of life, can work productively and is able to make a 
        contribution to their community.
        
        Common issues among students include:
        - Academic pressure and stress
        - Financial concerns
        - Social isolation and loneliness
        - Sleep problems
        - Anxiety and depression
        
        Universities should provide:
        - Counseling services with trained professionals
        - Mental health awareness campaigns
        - Peer support programs
        - Flexible academic accommodations
        - Crisis intervention services
        
        Recommended coping strategies for students:
        - Maintain a regular sleep schedule
        - Exercise regularly
        - Practice mindfulness and meditation
        - Build social connections
        - Seek help early when needed
        """,
        
        "University Counseling Best Practices": """
        Effective University Counseling Services
        
        Counseling centers should offer:
        - Individual therapy sessions
        - Group therapy for common issues
        - Workshops on stress management
        - Crisis intervention available 24/7
        - Referrals to specialized care when needed
        
        Signs that a student may need help:
        - Significant changes in academic performance
        - Social withdrawal and isolation
        - Changes in sleep or eating patterns
        - Expressions of hopelessness or worthlessness
        - Increased irritability or anger
        
        How to support a struggling student:
        - Express concern in a private setting
        - Listen without judgment
        - Encourage them to seek professional help
        - Offer to accompany them to counseling services
        - Follow up to check on their well-being
        """
    }
    
    # Create embeddings for the documents
    model = SentenceTransformer('all-MiniLM-L6-v2')
    documents = []
    embeddings = []
    
    for title, content in resources.items():
        # Split content into chunks (simplified approach)
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]
        for chunk in chunks:
            documents.append({"title": title, "content": chunk})
            embedding = model.encode(chunk)
            embeddings.append(embedding)
    
    # Create FAISS index for efficient similarity search
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return documents, index, model

# Retrieve relevant documents
def retrieve_documents(query, index, documents, model, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in indices[0]:
        if i < len(documents):
            results.append(documents[i])
    
    return results

# Generate response using Gemini
def generate_response(query, context_docs, gemini_model):
    if gemini_model is None:
        # Mock response if Gemini is not configured
        return "I understand you're asking about student mental health. Based on available resources, it's important to prioritize self-care, maintain social connections, and seek professional help when needed. Most universities offer counseling services that can provide personalized support."
    
    # Prepare context from retrieved documents
    context = "\n\n".join([f"From {doc['title']}:\n{doc['content']}" for doc in context_docs])
    
    prompt = f"""
    You are a mental health assistant for university students. Answer the question using only the provided context.
    If the answer isn't in the context, say you don't know rather than making up information.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"I encountered an error generating a response: {str(e)}. Please try again later."

# Main application
def main():
    # Setup
    gemini_model = setup_gemini()
    documents, index, embedding_model = load_documents()
    
    # Display UI
    st.subheader("Ask a Question")
    
    # Example questions
    example_questions = [
        "What are common mental health issues for students?",
        "How can universities support student mental health?",
        "What are signs that a student might be struggling?",
        "What coping strategies are recommended for students?"
    ]
    
    st.write("**Example questions you can try:**")
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(question, key=f"example_{i}"):
                st.session_state.current_question = question
    
    # Question input
    question = st.text_input(
        "Or type your own question:",
        value=st.session_state.get("current_question", ""),
        placeholder="e.g., How can I manage exam stress?"
    )
    
    if st.button("Get Answer") and question:
        with st.spinner("Searching resources and generating answer..."):
            # Retrieve relevant documents
            relevant_docs = retrieve_documents(question, index, documents, embedding_model)
            
            # Generate response
            response = generate_response(question, relevant_docs, gemini_model)
            
            # Display response
            st.subheader("Answer")
            st.write(response)
            
            # Show sources
            with st.expander("View sources used"):
                for i, doc in enumerate(relevant_docs):
                    st.write(f"**Source {i+1}: {doc['title']}**")
                    st.write(doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content'])
                    st.divider()
    
    # Resources section
    st.subheader("Available Resources")
    with st.expander("View mental health resources"):
        st.write("""
        This Q&A system is powered by the following verified mental health resources:
        
        1. **WHO Student Mental Health Guide** - Comprehensive guidelines on student mental health
        2. **University Counseling Best Practices** - Recommendations for effective campus counseling services
        
        In a full implementation, we would include:
        - University-specific mental health policies
        - Local counseling service information
        - Crisis hotline numbers
        - Self-help worksheets and resources
        """)
    
    # Disclaimer
    st.info("""
    **Disclaimer**: This is a demonstration system for educational purposes only. 
    It does not provide medical advice. If you're experiencing a mental health crisis, 
    please contact your campus counseling center or emergency services.
    """)

if __name__ == "__main__":
    main()