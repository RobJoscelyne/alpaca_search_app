import math
import re
import streamlit as st
from collections import Counter
from bs4 import BeautifulSoup
import nltk
from nltk.stem import PorterStemmer
import base64
import PIL.Image
from streamlit import cache


image = PIL.Image.open('llama.jpg')

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

image_width = int(image.size[0] * 0.45)
#st.image(image, width=image_width)

stemmer = PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')

# Define parameters
K1 = 2.6 # tuning parameter for term frequency scaling
B = 0.81 # tuning parameter for document length scaling
K3 = 0.0  # tuning parameter for query term frequency scaling
N = 10 # number of top results to show

stopwords_data = set(nltk.data.load('corpora/stopwords/english', format='raw').decode('utf-8').split())

def clean_text(text):
    # remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    if text is None:
        return None
    # remove non-alphanumeric characters
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    # convert to lowercase
    text = text.lower()
    # tokenize the text into individual words or subwords
    tokens = nltk.word_tokenize(text)
    # stem the tokens to their root form
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # remove stop words after stemming
    stopwords = stopwords_data
    stemmed_tokens = [token for token in stemmed_tokens if token not in stopwords]
    return stemmed_tokens



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


# read the XML file into a string variable
with open('cran.all.1400.xml', 'r') as f:
    data = f.read()

# parse the XML using BeautifulSoup
soup = BeautifulSoup(data, 'html.parser')

# extract the document tags from the XML
docs = soup.find_all('doc')

# clean the text data and extract titles and document IDs
docs_cleaned = []
titles = []
doc_ids = []
bibs = []
authors = []

for doc in docs:
    title_tag = doc.find('title')
    if not title_tag or not title_tag.string:
        continue

    title = title_tag.string.strip()
    titles.append(title)
    
    doc_id = doc.find('docno').string.strip()
    doc_ids.append(doc_id)

    text_tag = doc.find('text').string.strip()
    docs_cleaned.append(text_tag)

    authors_tag = doc.find('author')
    if authors_tag and authors_tag.string:
        author = authors_tag.string.strip()
    else:
        author = 'Unknown'  # Assign a default value for missing author names
    authors.append(author)

    bibs_tag = doc.find('bib')
    if bibs_tag and bibs_tag.string:
        bib = bibs_tag.string.strip()
    else:
        bib = 'Unknown'  # Assign a default value for missing author names
    bibs.append(bib)


# Add this function to create the inverted index
def create_inverted_index(docs):
    inverted_index = {}
    for i, doc in enumerate(docs):
        terms = clean_text(doc)
        for term in terms:
            if term not in inverted_index:
                inverted_index[term] = []
            inverted_index[term].append(i)
    return inverted_index

# Compute idfs
def get_inverse_doc_freqs(docs):
    """Get inverse document frequencies from a list of documents"""
    num_docs = len(docs) # total number of documents
    doc_freqs = Counter() # count document occurrences for each term
    for doc in docs:
        terms = set(clean_text(doc)) # get unique terms in document

        #terms = set(tokenize(doc)) # get unique terms in document
        doc_freqs.update(terms) # update document frequencies

    idfs = {} # store inverse document frequencies for each term
    for term, df in doc_freqs.items():
        idf = math.log(num_docs / df) # compute inverse document frequency
        idfs[term] = idf

    return idfs

# Compute BM25 score for a query-document pair
def get_bm25_score(query, doc, idfs):
    """Get BM25 score for a query-document pair"""
    #terms = tokenize(doc)
    terms = clean_text(doc)

    term_freqs = Counter(terms)
    doc_length = len(terms)
    query_terms = [stemmer.stem(term) for term in query]
    query_freqs = Counter(query_terms)
    score = 0

    for term in query_terms:
        if term not in idfs:
            continue
        term_freq = term_freqs[term] if term in term_freqs else 0
        query_freq = query_freqs[term] if term in query_freqs else 0
        score += idfs[term] * ((term_freq * (K1 + 1)) / (term_freq + K1 * (1 - B + B * (doc_length / N)))) * ((K3 + 1) * query_freq / (K3 + query_freq))
    return score


# Get inverse document frequencies
idfs = get_inverse_doc_freqs(docs_cleaned)

# Replace the search function with the updated version
def search(query):
    # Tokenize and stem the query
    query_terms = clean_text(query)

    # Compute BM25 scores for relevant documents using the inverted index
    relevant_docs = set()
    for term in query_terms:
        if term in inverted_index:
            relevant_docs.update(inverted_index[term])

    scores = []
    for i in relevant_docs:
        doc = docs_cleaned[i]
        score = get_bm25_score(query_terms, doc, idfs)
        scores.append((i, score))

    # Sort the documents by score and return the top N results
    top_results = sorted(scores, key=lambda x: x[1], reverse=True)[:N]
    return top_results

# Create the inverted index after cleaning the documents
inverted_index = create_inverted_index(docs_cleaned)


import time
import streamlit as st


st.image(image, width=400)

query = st.text_input('Enter your query:')

if query:
    start_time = time.time()  # measure start time
    results = search(query)
    end_time = time.time()  # measure end time
    search_time = round(end_time - start_time, 3)  # compute search time
    st.write(f'Search completed in {search_time} seconds')

    for i, score in results[:N]:
        col1, col2, col3 = st.columns([1, 3, 3])  # Divide the row into 3 columns
        with col1:
            st.write(f"Doc ID: {doc_ids[i]}")
        with col2:
            clean_title = titles[i].capitalize().rstrip('.')  # Remove period at the end of the title
            st.write(f"Title: {clean_title}")
        with col3:
            with st.expander("View"):
                st.markdown(display_document(doc_ids[i], titles[i], authors[i], bibs[i], docs_cleaned[i]), unsafe_allow_html=True)





