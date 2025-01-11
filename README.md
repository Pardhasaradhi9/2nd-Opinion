# 2nd Opinion - Your Personal Shopping Assistant

2nd Opinion is an AI-powered 🤖 application designed to help users make informed purchasing decisions 🛒 by analyzing product reviews and providing personalized recommendations. Whether you're buying a 💻, 📺, 📱, or any other product, 2nd Opinion has your back!

## Features
- **Review Aggregation**: Scrapes and aggregates reviews from Amazon for multiple products using a custom-built **web scraper**.
- **Summarization**: Provides a concise summary of each product using a **deep learning-based summarizer** that processes customer reviews.
- **AI Assistant (Peggy)**: Powered by **hybrid search (Advanced RAG)**, Peggy answers user questions, compares products, and helps users make decisions.

## How It Works
1. Input the URLs of the products you're considering.
2. 2nd Opinion scrapes and aggregates reviews for all products.
3. Get a concise summary of each product based on customer feedback.
4. Chat with **Peggy**, the AI assistant, to ask questions, compare products, and make decisions.

## Technologies Used
- **Web Scraping**: Custom-built scraper to extract reviews from Amazon.
- **Summarization**: Deep learning models (e.g., Hugging Face Transformers) to generate concise summaries of product reviews.
- **AI Assistant (Peggy)**:
  - **Hybrid Search**: Combines **vector-based retrieval** (using embeddings) and **keyword-based retrieval** (BM25) for accurate and relevant results.
  - **Advanced RAG (Retrieval-Augmented Generation)**: Enhances Peggy's ability to provide well-reasoned and context-aware responses.
- **Streamlit**: For building the intuitive and user-friendly interface.

## Demo
Watch the demo video [here](demo.mp4).

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Pardhasaradhi9/2nd-Opinion.git

## Like the work then give a star when used. Thank You 🙏🏾 ✌🏽
