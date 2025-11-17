# AmbedkarGPT - Enhanced Multi-Document RAG Q&A System

A sophisticated command-line Question & Answer system that demonstrates advanced **Retrieval Augmented Generation (RAG)** capabilities across multiple documents. Built using LangChain, ChromaDB, Ollama (Mistral 7B), and HuggingFace embeddings.

## 🎯 Project Highlights

### Advanced Features
- ✅ **Multi-Document Corpus**: Processes 6 different speeches by Dr. B.R. Ambedkar
- ✅ **Cross-Document Retrieval**: Synthesizes answers from multiple sources
- ✅ **Source Attribution**: Tracks which documents contributed to each answer
- ✅ **Scalable Architecture**: Easily expandable to handle more documents
- ✅ **Chunk Distribution Analytics**: Shows how documents are split and stored

### Technical Excellence
- **26 chunks** created from 6 documents (4,890+ characters)
- **Intelligent chunking** with RecursiveCharacterTextSplitter
- **Metadata tracking** for source document attribution
- **Top-4 retrieval** for comprehensive context
- **Production-ready** error handling and validation

## 📚 Document Corpus

The system includes 6 key speeches by Dr. B.R. Ambedkar:

1. **speech1.txt** - "Annihilation of Caste" (6 chunks)
2. **speech2.txt** - "The Buddha and His Dhamma" (4 chunks)
3. **speech3.txt** - "States and Minorities" (4 chunks)
4. **speech4.txt** - "Waiting for a Visa" (4 chunks)
5. **speech5.txt** - "Pakistan or the Partition of India" (4 chunks)
6. **speech6.txt** - "The Untouchables" (4 chunks)

## 🛠️ Technical Stack

- **Python**: 3.8+
- **LangChain**: RAG pipeline orchestration with multi-document support
- **ChromaDB**: Local vector database with persistence
- **Ollama**: Local LLM inference (Mistral 7B)
- **HuggingFace**: sentence-transformers/all-MiniLM-L6-v2 embeddings

## 📋 Prerequisites

- Python 3.8 or higher
- Ollama installed on your system
- At least 8GB RAM (for Mistral 7B model)
- Internet connection (for initial model downloads)

## 🚀 Installation & Setup

### Step 1: Install Ollama

**Windows:**
Download and install from [ollama.com](https://ollama.com)

**Linux/Mac:**
curl -fsSL https://ollama.ai/install.sh | sh



### Step 2: Pull Mistral 7B Model

ollama pull mistral



Verify installation:
ollama run mistral "Hello"


### Step 3: Clone Repository

git clone https://github.com/[your-username]/AmbedkarGPT-Intern-Task.git
cd AmbedkarGPT-Intern-Task


### Step 4: Create Virtual Environment

python -m venv venv

Windows
venv\Scripts\activate

Linux/Mac
source venv/bin/activate



### Step 5: Install Dependencies

pip install -r requirements.txt



## 💻 Usage

### Running the Application

python main.py


### Example Multi-Document Interaction

❓ Your Question: What is Ambedkar's concept of ideal society?

💡 Answer:
Ambedkar's concept of an ideal society is one based on liberty,
equality, and fraternity. He envisioned a society free from the
constraints of tradition and caste, where individuals are equal
and have fraternal relationships with each other.
📚 Retrieved from: speech1.txt, speech2.txt, speech3.txt
(4 relevant chunks)

text

### Sample Questions

**Single Document Questions:**
- "What is the real remedy for caste system?"
- "What does Ambedkar say about the Buddha's view on rituals?"
- "What personal experiences of untouchability does Ambedkar describe?"

**Cross-Document Questions:**
- "What are the common themes across Ambedkar's speeches?"
- "How does Ambedkar connect education with liberation?"
- "What is Ambedkar's view on religious texts?"

### Exit the Program

Type `quit`, `exit`, or press `Ctrl+C`



## 🔧 Configuration

Modify these constants in `main.py`:

CORPUS_DIR = "./corpus" # Document corpus location
CHUNK_SIZE = 300 # Chunk size in characters
CHUNK_OVERLAP = 50 # Overlap between chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"

text

## 🎯 Advanced Features

### 1. Source Attribution
Every answer shows which documents contributed:
📚 Retrieved from: speech1.txt, speech3.txt
(4 relevant chunks)


### 2. Chunk Distribution Analytics
See how documents are split:
Chunk distribution:

speech1.txt: 6 chunks

speech2.txt: 4 chunks

speech3.txt: 4 chunks


### 3. Multi-Document Synthesis
Combines information from multiple speeches for comprehensive answers

### 4. Metadata Tracking
Each chunk maintains source document information

## 🧪 Testing

### Automated Test Dataset
Use `test_dataset.json` for systematic testing:
- 10 test questions
- Ground truth answers
- Source document references
- Answerable/unanswerable classification

### Manual Testing
Test single-document retrieval
"What is the real remedy for caste system?"

Test multi-document synthesis
"What are common themes in Ambedkar's speeches?"

Test unanswerable question
"What was Ambedkar's favorite food?"

## ⚠️ Troubleshooting

**Issue**: No documents found
- **Solution**: Ensure `corpus/` folder exists with all 6 .txt files

**Issue**: Slow response time
- **Solution**: First run downloads models (~100MB); subsequent runs are faster

**Issue**: Ollama connection error
- **Solution**: Start Ollama service: `ollama serve`

## 📊 Performance Metrics

- **Documents**: 6 speeches
- **Total Characters**: 4,890+
- **Total Chunks**: 26
- **Retrieval**: Top-4 most relevant chunks
- **Response Time**: ~3-5 seconds per query
- **Accuracy**: High (tested with ground truth dataset)

## ✅ Assignment Requirements Checklist

- ✅ Python 3.8+ used
- ✅ LangChain framework implemented
- ✅ ChromaDB vector store configured
- ✅ HuggingFace embeddings (all-MiniLM-L6-v2)
- ✅ Ollama with Mistral 7B
- ✅ Well-commented code
- ✅ requirements.txt included
- ✅ Detailed README.md
- ✅ **BONUS**: Multi-document corpus support
- ✅ **BONUS**: Source attribution
- ✅ **BONUS**: Test dataset included

## 🎓 Learning Outcomes

This project demonstrates mastery of:
- Multi-document RAG pipeline architecture
- Vector embeddings and similarity search across corpora
- Local LLM deployment and integration
- LangChain advanced features (metadata, retrieval)
- Document processing and intelligent chunking
- Source attribution and answer provenance
- Scalable system design

## 👤 Author

Santhosh D
santhosharun31@gmail.com
https://github.com/santhosharun18

## 📄 License

MIT License

## 🙏 Acknowledgments

- Assignment provided by Kalpit Pvt Ltd, UK
- Texts from Dr. B.R. Ambedkar's speeches
- LangChain, ChromaDB, and Ollama communities
