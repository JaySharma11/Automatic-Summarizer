import os
from werkzeug.utils import secure_filename

import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
import pandas as pd

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

from flask import Flask,render_template,request

def get_wiki_content(url):
    req_obj = requests.get(url)
    text = req_obj.text
    soup = BeautifulSoup(text, "html.parser")
    all_paras = soup.find_all("p")
    wiki_text = []
    curstr = ""
    for para in all_paras:
        wiki_text.append(para.text)
    return wiki_text

def top10_sent(url, top_n):
    stop_words = stopwords.words('english')
    summarize_text = []

    filedata = get_wiki_content(url)
    sentences = []

    for i in range(len(filedata)):
      article = filedata[i].split(". ")
      for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    try:
        sentences.pop()
    except IndexError as e:
        summarize_text.append("Input text is too short to generate summary.")
        return summarize_text

    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    # print("Indexes of top ranked_sentence order are ", ranked_sentence, "\n \n")
    try:
        for i in range(top_n):
            summarize_text.append(" ".join(ranked_sentence[i][1]))
    except IndexError as e:
        summarize_text.append("Input text is too short to generate summary.")

    # print("Summarize Text:- \n \n", " ".join(summarize_text))
    return summarize_text

def read_article(file_name,top_n):
    file = open(file_name, "r", encoding='utf-8')
    filedata = file.readlines()
    sentences = []

    for i in range(len(filedata)):
      article = filedata[i].split(". ")
      for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
      stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    # build the vector for the first sentence
    for w in sent1:
      if w in stopwords:
        continue
      vector1[all_words.index(w)] += 1


    # build the vector for the second sentence
    for w in sent2:
      if w in stopwords:
        continue
      vector2[all_words.index(w)] += 1

    np.seterr(divide='ignore')
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))

    for idx1 in range(len(sentences)):
      for idx2 in range(len(sentences)):
        if idx1 == idx2: #ignore if both are same sentences
          continue
        similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 — Read text anc split it
    sentences = read_article(file_name,top_n)

    # Step 2 — Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 — Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 — Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    # print("Indexes of top ranked_sentence order are ", ranked_sentence, "\n \n")
    try:
        for i in range(top_n):
            summarize_text.append(" ".join(ranked_sentence[i][1]))
    except IndexError as e:
        summarize_text.append("Input text is too short to generate summary.")

    # Step 5 — Offcourse, output the summarize texr
    # print("Summarize Text:- \n \n", " ".join(summarize_text))
    return summarize_text

def text_to_sum(text, top_n):
    stop_words = stopwords.words('english')
    summarize_text = []

    textarea = text.split('\n\n')

    filedata = textarea
    sentences = []

    for i in range(len(filedata)):
      article = filedata[i].split(". ")
      for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()

    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    # print("Indexes of top ranked_sentence order are ", ranked_sentence, "\n \n")
    try:
        for i in range(top_n):
            summarize_text.append(" ".join(ranked_sentence[i][1]))
    except IndexError as e:
        summarize_text.append("Input text is too short to generate summary.")
    # print("Summarize Text:- \n \n", " ".join(summarize_text))
    return summarize_text

app = Flask(__name__)

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        url = request.form.get("url")
        txtfile = request.files['txtfile']
        text = request.form.get("textarea")
        n = request.form.get("length")

        if len(n) > 0:
            top_n = int(n)
        else:
            summary_content = []
            summary_content.append("Please specify the length of summary")
            return render_template('data.html',form_data = summary_content)

        if len(url) > 0:
            summary_content = top10_sent(url, top_n)

        elif txtfile.filename != '':
            txtfile.save(secure_filename(txtfile.filename))
            summary_content = generate_summary(txtfile.filename, top_n)
            os.remove(txtfile.filename)

        elif len(text) > 0:
            summary_content = text_to_sum(text, top_n)

        else:
            summary_content = []
            summary_content.append("Please give input text to generate summary and Congratulations for generating the shortest summary possible.")

    return render_template('data.html',form_data = summary_content)



app.run(host='localhost', port=5000)
