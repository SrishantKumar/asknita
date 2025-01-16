# AskNITA 🤖

AskNITA is an advanced AI-powered documentation assistant that combines multiple language models to provide accurate and context-aware responses to your questions. Built with modern technologies like Streamlit and Supabase, it uses vector similarity search to intelligently process and retrieve information from documentation.

![AskNITA Banner](https://raw.githubusercontent.com/SrishantKumar/asknita/main/docs/banner.png)

## 🌟 Key Features

- 🧠 **Multi-Model AI Integration**
  - Mistral AI for primary responses
  - Groq for fast processing
  - OpenRouter for model variety
  - Local Ollama for offline capabilities

- 🔍 **Smart Search & Retrieval**
  - Vector-based similarity search
  - Context-aware response generation
  - Automatic chunk optimization

- 🔄 **Reliability Features**
  - Automatic failover between AI providers
  - Rate limit handling
  - Error recovery mechanisms

- 🛠 **Developer Tools**
  - Documentation crawler and indexer
  - Vector database integration
  - Easy-to-use Streamlit interface

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Supabase account
- API keys for:
  - Mistral AI
  - Groq
  - OpenRouter
- Local Ollama installation

### Installation

1. Clone the repository:
```bash
git clone https://github.com/SrishantKumar/asknita.git
cd asknita
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

### Running the Application

1. Start the UI:
```bash
streamlit run asknita_ui.py
```

2. Index new documentation:
```bash
python crawl_asknita.py
```

## 🏗 Architecture

### Components

- **Frontend**: Streamlit-based interactive UI
- **Backend**: 
  - Multiple AI model integrations
  - Vector database (Supabase)
  - Documentation crawler
- **Database**: PostgreSQL with vector extensions

### AI Provider Integration

```python
AI_PROVIDERS = {
    'primary': 'Mistral AI',
    'backup': ['Groq', 'OpenRouter'],
    'local': 'Ollama'
}
```

## 📦 Project Structure

```
asknita/
├── asknita_ui.py      # Main Streamlit interface
├── crawl_asknita.py   # Documentation crawler
├── site_pages.sql     # Database schema
├── requirements.txt   # Dependencies
└── .env              # Configuration
```

## 🛠 Configuration

### Environment Variables

```env
MISTRAL_API_KEY=your_mistral_key
GROQ_API_KEY=your_groq_key
OPENROUTER_API_KEY=your_openrouter_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_key
```

### Database Setup

1. Create vector-enabled database:
```sql
-- Execute commands in site_pages.sql
```

2. Initialize tables and indexes:
```bash
python setup_database.py
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Supabase](https://supabase.io/)
- AI models from:
  - [Mistral AI](https://mistral.ai/)
  - [Groq](https://groq.com/)
  - [OpenRouter](https://openrouter.ai/)
  - [Ollama](https://ollama.ai/)
