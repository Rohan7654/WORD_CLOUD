import PyPDF4
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from wordcloud import WordCloud
from string import punctuation
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
pdfReader = PyPDF4.PdfFileReader(open('doc.pdf', 'rb'))
from nltk.corpus import stopwords

##print(pdfReader.numPages)

text = ''

for page in pdfReader.pages:
    text += page.extractText()

##print(text)

def remove_num(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

text = np.vectorize(remove_num)(text)


def remove_punct(text):
    text = ' '.join(word.strip(punctuation) for word in text.split() if word.strip(punctuation))
    return text

text = np.vectorize(remove_punct)(text)

def remove_u(text):
    text = text.replace('_','')
    text = text.replace('?','')
    text = text.replace('•','')
    text = text.replace("@",'')
    text = text.replace('▯','')
    text = text.replace("'",'')
    text = text.replace(",","")
    return text

text = np.vectorize(remove_u)(text)

def remove_extra_space(text):
    word_list = text.split()
    text = ' '.join(word_list)
    return text

text = np.vectorize(remove_extra_space)(text)

##stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the","Mr", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    word_list = text.split()
    word_list = [word for word in word_list if word not in stop_words]
    text = ' '.join(word_list)
    return text

text = np.vectorize(remove_stopwords)(text)
text = text.tolist()

##print(text)

mask = np.array(Image.open(r'C:\Users\Rohan\Desktop\kntka.png'))
plt.imshow(mask)
plt.axis("off")

wordcloud = WordCloud(mask=mask, width=3000, height=1000,contour_color="white", max_words=10000,relative_scaling = 0, background_color = "black").generate(text)

image_colors = ImageColorGenerator(mask)

plt.figure(figsize=[20,15])
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("assignment.png")
img = cv2.imread("assignment.png")
cv2.imshow("out",img)

##print("done")
