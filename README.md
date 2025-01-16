# AskNITA: Multi-Model AI Documentation Assistant

An intelligent documentation assistant that leverages multiple AI models (Mistral, Groq, OpenRouter, Ollama) to provide accurate and reliable answers to your questions. Built with Streamlit and Supabase, AskNITA uses vector similarity search to find and analyze relevant documentation chunks before generating responses.

## Features

- **Multiple AI Providers**: Uses Mistral, Groq, OpenRouter, and Ollama for robust and reliable responses
- **Vector Database Storage**: Utilizes Supabase for efficient storage and retrieval of documentation chunks
- **Smart Failover**: Automatically switches between AI providers if one fails or hits rate limits
- **Documentation Crawler**: Built-in crawler to index and chunk documentation content
- **Modern Streamlit UI**: Clean and intuitive user interface for asking questions
- **Vector Similarity Search**: Finds the most relevant documentation chunks for accurate answers

## Prerequisites

- Python 3.11+
- Supabase account and database
- API keys for:
  - Mistral AI
  - Groq
  - OpenRouter
  - Running Ollama instance (local)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd askNITA
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file with your API keys and configuration:
```env
MISTRAL_API_KEY=your_mistral_api_key
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_key
```

## Usage

1. Start the Streamlit interface:
```bash
streamlit run asknita_ui.py
```

2. To crawl and index new documentation:
```bash
python crawl_asknita.py
```

## Database Setup

Execute the SQL commands in `site_pages.sql` to:
1. Create the necessary tables
2. Enable vector similarity search
3. Set up proper indexing

## Project Structure

- `asknita_ui.py`: Main Streamlit UI application
- `crawl_asknita.py`: Documentation crawler and indexer
- `site_pages.sql`: Database schema and setup
- `requirements.txt`: Project dependencies
- `.env`: Configuration file

## Features in Detail

### AI Provider Management
- Automatic failover between different AI providers
- Rate limit handling and error recovery
- Configurable retry mechanisms

### Vector Search
- Semantic similarity search using embeddings
- Configurable number of relevant chunks
- Context-aware response generation

### Documentation Processing
- Automatic chunking of documentation
- Preservation of code blocks and formatting
- Metadata extraction and storage

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]
