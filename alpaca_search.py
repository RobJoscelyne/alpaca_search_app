# Import necessary libraries and modules
import math
import re
import streamlit as st
from collections import Counter
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer
import PIL.Image

# Load an image for display in the Streamlit app
image = PIL.Image.open('logo.jpg')

# Apply custom CSS styles to the Streamlit app
st.write(
    """
<style>
    .stApp {
        background-color: white;
    }
    .stTextInput input {
        background-color: lightgrey;
    }
    .stApp * {
        color: black;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize the stemmer for word stemming
stemmer = PorterStemmer()

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# Define BM25 parameters
K1 = 2.6
B = 0.81
K3 = 0.0
N = 10  # Number of top results to show

# Load stopwords for text processing
stopwords_data = set(nltk.data.load('corpora/stopwords/english', format='raw').decode('utf-8').split())

# Function to clean and preprocess the text
def clean_text(text):
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    # Remove non-alphanumeric characters
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    # Convert to lowercase and tokenize
    tokens = nltk.word_tokenize(text.lower())
    # Stem tokens and remove stopwords
    return [stemmer.stem(token) for token in tokens if token not in stopwords_data]

# Function to display document details in the Streamlit app
def display_document(doc_id, title, author, bib, text):
    content = f"""
    <div style="width: 100%;">
        <p><strong>Doc ID:</strong> {doc_id}</p>
        <p><strong>Title:</strong> {title}</p>
        <p><strong>Author:</strong> {author}</p>
        <p><strong>Bib:</strong> {bib}</p>
        <h3>Document Text</h3>
        <p>{text}</p>
    </div>
    """
    return content

# Parse the XML data to extract document details
with open('cran.all.1400.xml', 'r') as f:
    soup = BeautifulSoup(f.read(), 'html.parser')

docs = soup.find_all('doc')
docs_cleaned, titles, doc_ids, bibs, authors = [], [], [], [], []

for doc in docs:
    title_element = doc.find('title')
    title = title_element.string.strip() if title_element and title_element.string else 'Unknown'
    titles.append(title)

    doc_id_element = doc.find('docno')
    doc_id = doc_id_element.string.strip() if doc_id_element and doc_id_element.string else 'Unknown'
    doc_ids.append(doc_id)

    text_element = doc.find('text')
    doc_text = text_element.string.strip() if text_element and text_element.string else 'No text available'
    docs_cleaned.append(doc_text)

    author_element = doc.find('author')
    author = author_element.string.strip() if author_element and author_element.string else 'Unknown'
    authors.append(author)

    bib_element = doc.find('bib')
    bib = bib_element.string.strip() if bib_element and bib_element.string else 'Unknown'
    bibs.append(bib)

# Create an inverted index for efficient search
def create_inverted_index(docs):
    inverted_index = {}
    for i, doc in enumerate(docs):
        for term in clean_text(doc):
            inverted_index.setdefault(term, []).append(i)
    return inverted_index

# Compute inverse document frequencies for terms
def get_inverse_doc_freqs(docs):
    num_docs = len(docs)
    doc_freqs = Counter(term for doc in docs for term in set(clean_text(doc)))
    return {term: math.log(num_docs / df) for term, df in doc_freqs.items()}

# Compute BM25 score for a query-document pair
def get_bm25_score(query, doc, idfs):
    terms = clean_text(doc)
    term_freqs = Counter(terms)
    query_terms = clean_text(query)
    score = 0
    for term in query_terms:
        if term not in idfs:
            continue
        term_freq = term_freqs.get(term, 0)
        score += idfs[term] * ((term_freq * (K1 + 1)) / (term_freq + K1 * (1 - B + B * (len(terms) / N))))
    return score

# Search function using BM25 scoring
def search(query):
    query_terms = clean_text(query)
    relevant_docs = {i for term in query_terms for i in inverted_index.get(term, [])}
    scores = [(i, get_bm25_score(query, docs_cleaned[i], idfs)) for i in relevant_docs]
    return sorted(scores, key=lambda x: x[1], reverse=True)[:N]

# Create the inverted index and compute idfs
inverted_index = create_inverted_index(docs_cleaned)
idfs = get_inverse_doc_freqs(docs_cleaned)

# Streamlit app interface
st.image(image, width=400)
query = st.text_input('Enter your query:')
if query:
    results = search(query)
    st.write(f'Search completed in {time.time() - start_time:.3f} seconds')
    for i, score in results:
        col1, col2, col3 = st.columns([1, 3, 3])
        with col1:
            st.write(f"Doc ID: {doc_ids[i]}")
        with col2:
            st.write(f"Title: {titles[i].capitalize().rstrip('.')}")
        with col3:
            with st.expander("View"):
                st.markdown(display_document(doc_ids[i], titles[i], authors[i], bibs[i], docs_cleaned[i]), unsafe_allow_html=True)
