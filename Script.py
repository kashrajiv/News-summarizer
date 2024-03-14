import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
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
               

sub = """Senate Majority Leader Chuck Schumer on Thursday criticized Israel Prime Minister Benjamin Netanyahu’s government, calling for new elections in a speech on the Senate floor on the Israel-Hamas war.

“As a lifelong supporter of Israel, it has become clear to me: The Netanyahu coalition no longer fits the needs of Israel after October 7. The world has changed, radically, since then, and the Israeli people are being stifled right now by a governing vision that is stuck in the past,” said Schumer, the highest-ranking Jewish elected official in America.

“Five months into this conflict, it is clear that Israelis need to take stock of the situation and ask: must we change course?” he continued. “At this critical juncture, I believe a new election is the only way to allow for a healthy and open decision-making process about the future of Israel, at a time when so many Israelis have lost their confidence in the vision and direction of their government.”

The humanitarian crisis impacting Palestinian civilians in Gaza has grown increasingly dire as Israel’s war against Hamas continues, a situation that has increased pressure on Democratic party officials, including President Joe Biden, to take a harder line against Israel. Congressional aid to Israel has stalled after the Senate passed a package with aid for Israel and Ukraine that has not been taken up in the House.

Netanyahu’s Likud party responded in a statement that Schumer is “expected to respect Israel’s elected government and not undermine it.”

“Israel is not a banana republic, but an independent and proud democracy that elected Prime Minister Netanyahu,” the statement said. “Contrary to Schumer’s words, the Israeli public supports a total victory over Hamas” and “opposes the return of the Palestinian Authority to Gaza.”

Schumer, who noted in his speech that he is the first Jewish Senate majority leader, condemned the Hamas terror attack on Israel on October 7 in his remarks and said he is “working in every way I can to support the Biden administration as negotiations continue to free every last one of the hostages.”

“October 7 and the shameless response to support that terrorist attack by some in America and around the globe have awakened the deepest fears of the Jewish people — that our annihilation remains a possibility,” Schumer said.

Schumer also said his “heart also breaks at the loss of so many civilian lives in Gaza.”

“I am anguished that the Israeli war campaign has killed so many innocent Palestinians,” he continued. “I know that my fellow Jewish Americans feel this same anguish when they see the images of dead and starving children and destroyed homes.”

Senate GOP Leader Mitch McConnell responded critically to Schumer’s speech, saying in his own remarks on the Senate floor, “Israel is not a colony of America whose leaders serve at the pleasure of the party in power in Washington. Only Israel’s citizens should have a say in who runs their government.”

“It is grotesque and hypocritical for Americans who hyperventilate about foreign interference in our own democracy to call for the removal of a democratically elected leader of Israel,” McConnell said. “This is unprecedented. We should not treat fellow democracies this way at all.”

House Speaker Mike Johnson said that Schumer’s call for a new election in Israel is “highly inappropriate” and “just plain wrong – for an American leader to play such a divisive role in Israeli politics while our closest ally in the region is in an existential battle for its very survival. We need to be standing with Israel.”

Johnson told CNN that House Republican leadership is “considering” putting a standalone Israel bill on the floor under the normal process after it failed to pass through an expedited process earlier this year.

“This probably does change the calculation so we are considering that,” the Louisiana Republican said following Schumer’s remarks.

Schumer said in his speech that he believes that “a majority of the Israeli public will recognize the need for change, and I believe that holding a new election once the war starts to wind down would give Israelis an opportunity to express their vision for the post-war future.”"""


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