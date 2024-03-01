import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
data = pd.read_csv('AI_Human.csv')

# Convert the text to lowercase
data['text'] = data['text'].str.lower()

# Tokenize the text
data['tokenized_text'] = data['text'].apply(word_tokenize)

# Remove stop words
stop_words = set(stopwords.words('english'))
data['filtered_text'] = data['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])

# Print the preprocessed data
print(data[['text', 'filtered_text']].head())

# Write preprocessed data to new CSV file
data.to_csv('preprocessed_data.csv', index=False)