import os
import time
import pandas as pd
from tqdm import tqdm
import neattext as nt
import streamlit as st
from dotenv import load_dotenv
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from bs4 import BeautifulSoup
from selenium import webdriver

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline

# Load environment variables
load_dotenv()

# Define a Document class with both 'page_content' and 'metadata' attributes
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

# Preprocess text using neattext
def preprocess_with_neattext(text):
    text_frame = nt.TextFrame(text)
    text_frame.remove_multiple_spaces()
    text_frame.remove_html_tags()
    text_frame.remove_urls()
    text_frame.remove_non_ascii()
    text_frame.remove_userhandles()
    text_frame.remove_hashtags()
    text_frame.remove_emojis()
    return text_frame.text

# Load and preprocess data
def load_and_preprocess_data(csv_paths, names):
    clean_full_desc = ''
    for i, csv_path in enumerate(tqdm(csv_paths, desc="Processing CSV files")):
        laptop_df = pd.read_csv(csv_path)
        clean_full_desc = clean_full_desc + f'The Review of {names[i]} are as follows: \n'
        full_desc = ' '.join(laptop_df['review_text'].dropna().astype(str))
        clean_full_desc = clean_full_desc + preprocess_with_neattext(full_desc) + '\n'
    return clean_full_desc

# Initialize the ensemble retriever
def initialize_retriever(chunks, embeddings):
    # Create a Vector Store
    vectorstore = Chroma.from_documents(chunks, embeddings)
    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Initialize BM25 retriever
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 3

    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever

# Function to update the page number in the URL
def update_page_number(url, page_number):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    query_params['pageNumber'] = [str(page_number)]
    updated_query = urlencode(query_params, doseq=True)
    updated_url = urlunparse(parsed_url._replace(query=updated_query))
    return updated_url

# Function to scrape reviews for a single product
def scrape_reviews(driver, url, product_name):
    i = 0
    names = []
    ratings = []
    rating_dates = []
    titles = []
    reviews_text = []

    while url is not None:
        try:
            i = i + 1
            if i > 1:
                url = update_page_number(url, i)  # Update the URL with the correct pageNumber
            
            driver.get(url)
            time.sleep(5)  # Allow the page to load

            # Check if the login page is displayed
            if "ap/signin" in driver.current_url:
                st.sidebar.warning("""Oh-o Login page detected. Please log in manually within 25 seconds.
                                   Peggy suggests not save the credential. Just log in and continue.""")
                time.sleep(25)  # Wait for 15 seconds for manual login
                driver.get(url)  # Reload the page after login
                time.sleep(5)  # Allow the page to load

            # Parse the page with BeautifulSoup
            html_data = BeautifulSoup(driver.page_source, 'html.parser')
            reviews = html_data.find_all('li', {'data-hook': 'review'})

            # If no reviews are found, pause for 15 seconds for manual intervention
            if not reviews:
                st.sidebar.warning("""Peggy Needs your help! No reviews found. Pausing for 25 seconds for manual intervention...
                                   Peggy suggests not save the credential. Just log in and continue.""")
                time.sleep(25)  # Wait for 15 seconds for manual intervention
                driver.get(url)  # Reload the page
                time.sleep(5)  # Allow the page to load
                html_data = BeautifulSoup(driver.page_source, 'html.parser')
                reviews = html_data.find_all('li', {'data-hook': 'review'})

                # If still no reviews, skip this product
                if not reviews:
                    st.sidebar.error(f"No reviews found after retry for {product_name}. Skipping...")
                    break

            for review in reviews:
                name = review.find('span', {'class': 'a-profile-name'}).text
                names.append(name.strip())
                rating = review.find('span', {'class': 'a-icon-alt'}).text
                ratings.append(rating)
                rating_date = review.find('span', {'data-hook': 'review-date'}).text
                rating_dates.append(rating_date)
                title = review.find('a', {'data-hook': 'review-title'}).text
                titles.append(title)
                review_text = review.find('span', {'data-hook': 'review-body'}).text
                reviews_text.append(review_text)

            # Check for next page
            url_check = html_data.find('li', {'class': 'a-last'})
            if url_check is None or url_check.find('a') is None:
                url = None
                
        except Exception as e:
            st.sidebar.error(f"Error occurred on page {i} for {product_name}: {str(e)}")
            url = None

    st.sidebar.success(f"Total reviews collected for {product_name}: {len(names)}")

    # Create DataFrame
    data = pd.DataFrame({
        'profile_name': names,
        'rating': ratings,
        'rating_date': rating_dates,
        'title': titles,
        'review_text': reviews_text
    })
    data['product_name'] = product_name

    return data

# Function to generate summaries
def generate_summary(data, product_name):
    # Combine all rows of the 'review_text' column into a single string
    full_desc = ' '.join(data['review_text'].dropna().astype(str))
    
    # Preprocess the text
    clean_full_desc = preprocess_with_neattext(full_desc)
    clean_full_desc = clean_full_desc.replace('.', '.<eos>')
    clean_full_desc = clean_full_desc.replace('?', '?<eos>')
    clean_full_desc = clean_full_desc.replace('!', '!<eos>')
    
    # Split text into chunks for summarization
    max_chunk = 500
    sentences = clean_full_desc.split('<eos>')
    current_chunk = 0
    chunks = []
    
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))
    
    # Join words in each chunk to form sentences
    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])
    
    # Summarize each chunk
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0)
    res = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=140, min_length=30, do_sample=False)
        res.extend(summary)
    
    # Combine all summaries into a single text
    summary_text = ' '.join([summ['summary_text'] for summ in res])
    
    # Summarize the combined summary text for a final concise summary
    final_summary = summarizer(summary_text, max_length=300, min_length=100, do_sample=False)
    
    return final_summary[0]['summary_text']

# Streamlit app
def main():
    st.set_page_config(page_title="2nd Opinion", page_icon="💻")
    
    # Initialize session state
    if 'dataframes' not in st.session_state:
        st.session_state.dataframes = {}
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for user inputs
    with st.sidebar:
        st.header("Input Parameters")
        number_of_products = st.number_input("Enter the number of products:", min_value=1, step=1)

        # Initialize lists to store product URLs and names
        product_urls = []
        product_names = []

        # Loop to collect product URLs and names
        for i in range(number_of_products):
            st.subheader(f"Product {i+1}")
            url = st.text_input(f"Enter the Amazon URL of product {i+1}:", key=f"url_{i}")
            name = st.text_input(f"Enter your desired name for the product {i+1}:", key=f"name_{i}")
            if url:
                review_url = url.replace("dp", "product-reviews") + "?pageNumber=1"
                product_urls.append(review_url)
                product_names.append(name)

        # Button to start scraping
        if st.button("Start Scraping"):
            if not product_urls:
                st.error("Please enter at least one product URL.")
            else:
                # Initialize WebDriver (e.g., Chrome)
                driver = webdriver.Chrome()

                for url, product_name in zip(product_urls, product_names):
                    st.sidebar.subheader(f"Scraping reviews for {product_name}...")
                    data = scrape_reviews(driver, url, product_name)
                    
                    # Save the dataframe to session state
                    st.session_state.dataframes[product_name] = data

                    # Generate and save the summary
                    summary = generate_summary(data, product_name)
                    st.session_state.summaries[product_name] = summary

                    st.sidebar.success(f"Scraping and summarization completed for {product_name}.")

                # Close the WebDriver
                driver.quit()

    # Main screen layout
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("/Users/pardhasaradhichukka/Desktop/Project/Reboot/pracrtise/bot1.jpg", width=100)  
    with col2:
        st.title("2nd Opinion")
        st.write("Harness the Power of AI for making better purchasing decisions")

    # Display summaries and dataframes
    if st.session_state.dataframes:
        selected_product = st.selectbox("Select a product:", list(st.session_state.dataframes.keys()))

        # Display the head of the selected dataframe
        st.subheader(f"DataFrame for {selected_product} (First 5 Rows)")
        st.dataframe(st.session_state.dataframes[selected_product].head())

        # Display the summary
        st.subheader(f"Summary for {selected_product}")
        st.write(st.session_state.summaries[selected_product])

    # Chat interface
    st.header("Chat Interface")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    # User input
    user_question = st.chat_input("Hey 👋🏽 I'm Peggy, I can help you with your product purchase. ")
    if user_question:
        # Add user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question, "avatar": "/Users/pardhasaradhichukka/Desktop/Project/Reboot/pracrtise/user.png"})  # Replace with your user avatar path

        # Display user question
        with st.chat_message("user", avatar="/Users/pardhasaradhichukka/Desktop/Project/Reboot/pracrtise/user.png"):  # Replace with your user avatar path
            st.write(user_question)

        HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
        )
        
        # Create chunks from the data
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents([Document(page_content=st.session_state.dataframes[selected_product]['review_text'].str.cat(sep=' '))])

        ensemble_retriever = initialize_retriever(chunks, embeddings)

        # Initialize LLM
        llm = HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-beta",
            model_kwargs={"temperature": 0.3, "max_new_tokens": 1024},
            huggingfacehub_api_token=HF_TOKEN,
        )
        # Define prompt template
        template = """
        <|system|>>
        You are Peggy, an AI assistant focused on providing accurate, well-reasoned responses based on given context. Your key characteristics are:
        - Analytical and methodical in approach
        - Clear and precise in communication
        - Detail-oriented while remaining accessible
        - Professional yet warm in tone

        Instructions:
        1. First, determine if the user's message is:
            a) A general greeting or conversation starter or farewell(like "hi", "hello", "thanks", "how are you", "thanks", "bye" etc..)
            b) A context-related question or request

        2. For general greetings/conversation:
            - Respond with a friendly and engaging greeting
            - If the user ends the conversation, respond with a polite farewell
            - If the user asks about your capabilities, respond with "I'm an AI assistant designed to answer questions based on the provided context. I'm here to help with that!"


        3. For context-related questions:
            - Carefully analyze the provided context
            - Break down the user's query into key components
            - Identify relevant information from the context
            - Use the identified information to form a well-reasoned response
            - Ensure the response is clear, accurate, and well-structured
            - If the context does not contain enough information to answer the query, respond with "I'm unable to find an answer to that question based on the provided context."
        
        You will get a $100 tip if you provide correct answer.

        CONTEXT: {context}
        </s>
        <|user|>
        {query}
        </s>
        <|assistant|>
        """

        prompt = ChatPromptTemplate.from_template(template)
        output_parser = StrOutputParser()

        # Define the chain
        chain = (
            {"context": ensemble_retriever, "query": RunnablePassthrough()}
            | prompt
            | llm
            | output_parser
        )
        # Generate bot response
        with st.spinner("Generating answer..."):
            response = chain.invoke(user_question)
            bot_response = response.split("<|assistant|>")[-1].strip()

        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response, "avatar": "/Users/pardhasaradhichukka/Desktop/Project/Reboot/pracrtise/bot.png"})  # Replace with your bot avatar path

        # Display bot response
        with st.chat_message("assistant", avatar="/Users/pardhasaradhichukka/Desktop/Project/Reboot/pracrtise/bot.png"):  # Replace with your bot avatar path
            st.write(bot_response)

        # Optionally, show full info in the sidebar
        with st.sidebar:
            with st.expander("Show Full Info"):
                st.write("### Full Info:")
                st.write(response)

if __name__ == "__main__":
    main()