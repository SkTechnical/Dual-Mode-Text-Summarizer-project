# ========== Required Installations ==========
!pip install -q pandas sentence-transformers transformers sentencepiece beautifulsoup4 requests

# ========== Imports ==========
import pandas as pd
import numpy as np
import nltk
import re
import networkx as nx
from nltk.tokenize import PunktSentenceTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from bs4 import BeautifulSoup
import requests
import torch

nltk.download('punkt', quiet=True)

# ========== Improved Web Scraper ==========
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            print("Main content section not found.")
            return ""
        for tag in content_div(['script', 'style', 'table', 'sup']):
            tag.decompose()
        paragraphs = content_div.find_all('p')
        text = ' '.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
        return text
    except Exception as e:
        print(f"Error fetching URL: {e}")
        return ""

# ========== Summarization Helpers ==========
def tokenize_sentences(text):
    tokenizer = PunktSentenceTokenizer()
    return [s for s in tokenizer.tokenize(text) if len(s.strip()) > 10], tokenizer

def get_bert_embeddings(sentences, model):
    return model.encode(sentences, show_progress_bar=True)

def build_similarity_matrix(vectors):
    sim_mat = cosine_similarity(vectors)
    np.fill_diagonal(sim_mat, 0)
    return sim_mat

def calculate_pagerank(sim_mat):
    graph = nx.from_numpy_array(sim_mat)
    return nx.pagerank(graph)

def get_ranked_sentence_map(sentences, pagerank_scores):
    return {sentences[i]: score for i, score in pagerank_scores.items()}

def generate_extractive_summary(text, tokenizer, sentence_rank_map, top_n=3):
    article_sents = tokenizer.tokenize(text)
    ranked_article_sents = sorted(
        [(sentence_rank_map.get(s, 0), s) for s in article_sents],
        reverse=True
    )
    summary_sents = [s for score, s in ranked_article_sents[:top_n]]
    return " ".join(summary_sents)

def generate_abstractive_summary(text, model, tokenizer):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    summary_ids = model.generate(
        inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ========== Save Summary to File ==========
def save_summary_to_file(original, extractive, abstractive, filename="summary_output.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("========== ORIGINAL TEXT ==========\n\n")
        f.write(original[:3000] + "...\n\n" if len(original) > 3000 else original + "\n\n")
        f.write("========== EXTRACTIVE SUMMARY (BERT) ==========\n\n")
        f.write(extractive + "\n\n")
        f.write("========== ABSTRACTIVE SUMMARY (T5) ==========\n\n")
        f.write(abstractive + "\n\n")
    print(f"\n✅ Summary saved to file: {filename}")

# ========== Main Function ==========
def summarize_input():
    user_input = input("Enter a website URL or paste your text:\n").strip()

    # Check if URL or raw text
    if user_input.startswith("http://") or user_input.startswith("https://"):
        print("\nFetching and summarizing content from website...\n")
        text = extract_text_from_url(user_input)
    else:
        print("\nSummarizing provided text...\n")
        text = user_input

    if len(text.strip()) < 100:
        print("Text is too short to summarize meaningfully.")
        return

    # Load models
    print("Loading models...")
    extractive_model = SentenceTransformer('all-MiniLM-L6-v2')
    abstractive_tokenizer = T5Tokenizer.from_pretrained('t5-base')
    abstractive_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    print("Models loaded.\n")

    # Extractive
    sentences, tokenizer = tokenize_sentences(text)
    bert_embeddings = get_bert_embeddings(sentences, extractive_model)
    sim_matrix = build_similarity_matrix(bert_embeddings)
    pagerank_scores = calculate_pagerank(sim_matrix)
    sentence_rank_map = get_ranked_sentence_map(sentences, pagerank_scores)
    extractive_summary = generate_extractive_summary(text, tokenizer, sentence_rank_map)

    # Abstractive
    abstractive_summary = generate_abstractive_summary(text, abstractive_model, abstractive_tokenizer)

    # Print
    print("\n========== ORIGINAL TEXT ==========\n")
    print(text[:1500] + "..." if len(text) > 1500 else text)
    print("\n========== EXTRACTIVE SUMMARY (BERT) ==========\n")
    print(extractive_summary)
    print("\n========== ABSTRACTIVE SUMMARY (T5) ==========\n")
    print(abstractive_summary)

    # Save to file
    save_summary_to_file(text, extractive_summary, abstractive_summary)

# ========== Run ==========
summarize_input()

from google.colab import files
files.download("summary_output.txt")