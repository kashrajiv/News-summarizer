import nltk
import re
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import numpy as np
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import pipeline

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
               
file = open("C:\\Users\\rjalakam\\Documents\\Projects\\News-summarizer\\Article.txt", encoding="utf8")
# h = file.read()
# print(h)

sub = file.read()
file.close()


subtitle = sub.replace("\n","")
sentences = sent_tokenize(subtitle)

print("Actual script: ", "\n", subtitle)

organized_sent = {k:v for v,k in enumerate(sentences)}

tf_idf = TfidfVectorizer(min_df=2, 
    strip_accents='unicode',
    max_features=None,
    lowercase = True,
    token_pattern=r'w{1,}',
    ngram_range=(1, 3), 
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True)

sentence_vectors = tf_idf.fit_transform(sentences)
sent_scores = np.array(sentence_vectors.sum(axis=1)).ravel()

N = 3
top_n_sentences = [sentences[index] for index in np.argsort(sent_scores, axis=0)[::-1][:N]]

# mapping the scored sentences with their indexes as in the subtitle
mapped_sentences = [(sentence,organized_sent[sentence]) for sentence in top_n_sentences]
# Ordering the top-n sentences in their original order
mapped_sentences = sorted(mapped_sentences, key = lambda x: x[1])
ordered_sentences = [element[0] for element in mapped_sentences]
# joining the ordered sentence
summary = " ".join(ordered_sentences)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

input_tensor = tokenizer.encode( subtitle, return_tensors="pt", max_length=512)

outputs_tensor = model.generate(input_tensor, max_length=160, min_length=120, length_penalty=2.0, num_beams=4, early_stopping=True)

print("YouTube video Summary:", "\n", tokenizer.decode(outputs_tensor[0]))

# summarizer = pipeline('summarization')

# summary = summarizer(subtitle, max_length = 180, min_length =  30)

# print(summary)