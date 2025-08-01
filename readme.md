# AI Chatbot with Web Search Integration

An intelligent chatbot that combines the power of Llama 3.2-1B language model with real-time web search capabilities. The system automatically decides when to search the internet for current information and when to rely on the AI's existing knowledge.

## 🌟 Features

- **Smart AI Responses**: Powered by Meta's Llama 3.2-1B Instruct model
- **Intelligent Web Search**: Automatically searches the web when current information is needed
- **Real-time Streaming**: See responses generated word-by-word as you chat
- **Source Citations**: Get links to sources when web search is used
- **Simple Interface**: Clean, intuitive chat interface built with Streamlit
- **Local Deployment**: Run everything on your own machine for privacy

## 🏗️ Architecture

- **Frontend**: Streamlit web interface
- **Backend**: FastAPI server with OpenAI-compatible API
- **AI Model**: Llama 3.2-1B Instruct (local deployment)
- **Search**: Serper.dev API for web search
- **Streaming**: Real-time token streaming for responsive chat experience

## 🚀 Quick Setup

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Get API Keys

#### Hugging Face Token (Required)
1. Go to [Hugging Face](https://huggingface.co/)
2. Create an account and get your access token
3. Go to Settings → Access Tokens → Create new token

#### Serper API Key (Required for web search)
1. Visit [Serper.dev](https://serper.dev/)
2. Sign up for a free account
3. Get your API key from the dashboard

### 5. Set Environment Variables

Create a `.env` file in the project root:

```bash
HUGGINGFACE_TOKEN=your_huggingface_token_here
SERPER_API_KEY=your_serper_api_key_here
```

### 6. Start the Backend Server

```bash
python llama_api.py
```

The FastAPI server will start at `http://127.0.0.1:8000`

### 7. Launch the Frontend

Open a new terminal window and run:

```bash
streamlit run frontend.py
```

The chat interface will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
ai-chatbot-web-search/
├── llama_api.py            # FastAPI backend server
├── frontend.py             # Streamlit frontend
├── search_service.py       # Web search logic
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
└── README.md               # This file
```

## 🔧 Configuration

### Model Settings
The system uses Llama 3.2-1B by default. You can modify model parameters in `llama_api.py`:

```python
generation_kwargs = dict(
    max_new_tokens=4000,      # Maximum response length
    temperature=0.7,          # Response creativity
    top_p=0.95,              # Nucleus sampling
    repetition_penalty=1.1,   # Avoid repetition
)
```

### Search Settings
Customize search behavior in `search_service.py`:

```python
# Add your own search keywords
self.search_keywords = [
    "latest", "recent", "current", "today", "news", 
    # Add more keywords as needed
]
```

## 💡 Usage Examples

### General Questions (No Search)
- "Explain how photosynthesis works"
- "Write a short poem about winter"
- "What's the capital of France?"

### Questions That Trigger Search
- "What's the latest news about AI?"
- "Current Bitcoin price"
- "Weather forecast for today"
- "Recent developments in climate change"

### Force Search
Use the "Force Web Search" checkbox in the sidebar to search for any query, even general knowledge questions.
