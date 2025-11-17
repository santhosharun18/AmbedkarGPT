
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings
import os
import sys
import glob

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Configuration constants
CORPUS_DIR = "./corpus"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def load_documents_from_corpus(corpus_dir):
    """
    Load all text documents from the corpus directory.
    
    Args:
        corpus_dir (str): Path to the corpus directory
        
    Returns:
        list: List of loaded documents
    """
    print(f"📄 Loading documents from {corpus_dir}...")
    
    if not os.path.exists(corpus_dir):
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    # Get all .txt files in corpus directory
    txt_files = glob.glob(os.path.join(corpus_dir, "*.txt"))
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {corpus_dir}")
    
    print(f"   Found {len(txt_files)} document(s):")
    for file in txt_files:
        print(f"   - {os.path.basename(file)}")
    
    # Load all documents
    all_documents = []
    for file_path in txt_files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            docs = loader.load()
            # Add source filename to metadata
            for doc in docs:
                doc.metadata['source'] = os.path.basename(file_path)
            all_documents.extend(docs)
        except Exception as e:
            print(f"   ⚠️  Warning: Could not load {file_path}: {e}")
    
    total_chars = sum(len(doc.page_content) for doc in all_documents)
    print(f"✅ Loaded {len(all_documents)} document(s) with {total_chars} total characters")
    
    return all_documents

def split_documents(documents):
    """
    Split documents into manageable chunks.
    
    Args:
        documents (list): List of documents
        
    Returns:
        list: List of document chunks
    """
    print(f"✂️  Splitting documents into chunks...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"✅ Created {len(chunks)} chunks")
    
    # Show chunk distribution by source
    from collections import Counter
    source_counts = Counter(chunk.metadata.get('source', 'unknown') for chunk in chunks)
    print(f"   Chunk distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"   - {source}: {count} chunks")
    
    return chunks

def create_vector_store(chunks):
    """
    Create embeddings and store them in ChromaDB vector store.
    
    Args:
        chunks (list): List of document chunks
        
    Returns:
        Chroma: ChromaDB vector store instance
    """
    if not chunks:
        raise ValueError("Cannot create vector store with empty chunks.")
    
    print(f"🔢 Creating embeddings using {EMBEDDING_MODEL}...")
    
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Remove old ChromaDB if exists
    if os.path.exists(CHROMA_DB_DIR):
        import shutil
        print(f"   Removing existing ChromaDB at {CHROMA_DB_DIR}...")
        shutil.rmtree(CHROMA_DB_DIR)
    
    # Create and persist ChromaDB vector store
    print(f"💾 Storing embeddings in ChromaDB at {CHROMA_DB_DIR}...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    print("✅ Vector store created successfully")
    return vector_store

def setup_qa_chain(vector_store):
    """
    Set up the RetrievalQA chain with Ollama LLM.
    
    Args:
        vector_store (Chroma): ChromaDB vector store
        
    Returns:
        RetrievalQA: Configured QA chain
    """
    print(f"🤖 Initializing Ollama with {LLM_MODEL} model...")
    
    # Initialize Ollama LLM
    llm = Ollama(
        model=LLM_MODEL,
        temperature=0.3
    )
    
    # Create custom prompt template
    prompt_template = """Use the following context from Dr. B.R. Ambedkar's speeches to answer the question.
    If you cannot find the answer in the context, say "I cannot find this information in the provided documents."
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve top 4 chunks
    )
    
    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    print("✅ QA chain initialized successfully")
    return qa_chain

def run_qa_system():
    """
    Main function to run the Q&A system.
    """
    print("\n" + "="*70)
    print("🚀 AmbedkarGPT - Enhanced RAG-based Q&A System")
    print("   Multi-Document Corpus Support")
    print("="*70 + "\n")
    
    # Step 1: Load documents from corpus
    try:
        documents = load_documents_from_corpus(CORPUS_DIR)
    except Exception as e:
        print(f"❌ Error loading documents: {e}")
        print("\nPlease ensure:")
        print("1. Create a 'corpus' folder in the project directory")
        print("2. Add speech1.txt through speech6.txt in the corpus folder")
        sys.exit(1)
    
    # Step 2: Split documents
    chunks = split_documents(documents)
    
    if not chunks:
        print("❌ Error: No chunks were created from documents.")
        sys.exit(1)
    
    # Step 3: Create vector store
    try:
        vector_store = create_vector_store(chunks)
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")
        sys.exit(1)
    
    # Step 4: Setup QA chain
    qa_chain = setup_qa_chain(vector_store)
    
    print("\n" + "="*70)
    print("✨ System Ready! You can now ask questions about Dr. Ambedkar's speeches.")
    print("   Type 'quit' or 'exit' to stop the program.")
    print("="*70 + "\n")
    
    # Interactive Q&A loop
    while True:
        question = input("\n❓ Your Question: ").strip()
        
        # Exit condition
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using AmbedkarGPT!")
            break
        
        # Skip empty questions
        if not question:
            print("⚠️  Please enter a valid question.")
            continue
        
        # Get answer from QA chain
        print("\n🔍 Processing your question...")
        try:
            result = qa_chain.invoke({"query": question})
            answer = result['result']
            
            print("\n💡 Answer:")
            print("-" * 70)
            print(answer)
            print("-" * 70)
            
            # Show source documents
            if result.get('source_documents'):
                sources = set(doc.metadata.get('source', 'unknown') 
                            for doc in result['source_documents'])
                print(f"\n📚 Retrieved from: {', '.join(sorted(sources))}")
                print(f"   ({len(result['source_documents'])} relevant chunks)")
                
        except Exception as e:
            print(f"\n❌ Error processing question: {str(e)}")
            print("Please ensure Ollama is running: ollama serve")

if __name__ == "__main__":
    try:
        run_qa_system()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
