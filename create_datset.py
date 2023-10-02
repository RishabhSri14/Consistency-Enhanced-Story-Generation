import numpy as np # linear algebra
import pandas as pd

import os
for dirname, _, filenames in os.walk('/home/civic/anlp/project/writingPrompts'):
    # print(dirname, filenames)
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
DIR = "/home/civic/anlp/project/writingPrompts/"
data = [DIR+"train", DIR+"test", DIR+"valid"]
print(data)

TARGET_DIR = '/home/civic/anlp/project/'
target_data = [TARGET_DIR+"train", TARGET_DIR+"test", TARGET_DIR+"valid"]

import csv
from tqdm import tqdm_notebook as tqdm

NUM_WORDS = 1000
combined_data = []
name_id = 0
fp = open(data[name_id] + ".wp_source") 
ft = open(data[name_id] + ".wp_target") 
stories = ft.readlines()
prompts = fp.readlines()
assert len(prompts) == len(stories)
for i in range(len(stories)):
    prompt = prompts[i].rstrip()
    story = " ".join(stories[i].split()[0:NUM_WORDS])
    combined_data.append([prompt, story])
fp.close()
ft.close()
with open('combined_traindata.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Prompt', 'Story'])
    for row in combined_data:
        csv_writer.writerow(row)
print('CSV file created: combined_traindata.csv')

NUM_WORDS = 1000
combined_data = []
name_id = 1
fp = open(data[name_id] + ".wp_source") 
ft = open(data[name_id] + ".wp_target") 
stories = ft.readlines()
prompts = fp.readlines()
assert len(prompts) == len(stories)
for i in range(len(stories)):
    prompt = prompts[i].rstrip()
    story = " ".join(stories[i].split()[0:NUM_WORDS])
    combined_data.append([prompt, story])
fp.close()
ft.close()
with open('combined_testdata.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Prompt', 'Story'])
    for row in combined_data:
        csv_writer.writerow(row)
print('CSV file created: combined_testdata.csv')

NUM_WORDS = 1000
combined_data = []
name_id = 2
fp = open(data[name_id] + ".wp_source") 
ft = open(data[name_id] + ".wp_target") 
stories = ft.readlines()
prompts = fp.readlines()
assert len(prompts) == len(stories)
for i in range(len(stories)):
    prompt = prompts[i].rstrip()
    story = " ".join(stories[i].split()[0:NUM_WORDS])
    combined_data.append([prompt, story])
fp.close()
ft.close()
with open('combined_valdata.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Prompt', 'Story'])
    for row in combined_data:
        csv_writer.writerow(row)
print('CSV file created: combined_valdata.csv')

import csv
from rake_nltk import Rake
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

csv_file = 'combined_traindata.csv'

r = Rake()

triplets = []

summarizer = LexRankSummarizer()

with open(csv_file, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  
    for row in tqdm(csv_reader):
        prompt, story = row[0], row[1]

        r.extract_keywords_from_text(story)
        keywords = r.get_ranked_phrases()[:10] 

        parser = PlaintextParser.from_string(story, Tokenizer("english"))
        abstract_sentences = summarizer(parser.document, 3)  

        abstract = " ".join([str(sentence) for sentence in abstract_sentences])
        triplet = (prompt, " ".join(keywords), abstract)

        triplets.append(triplet)

new_csv_file = 'new_traindata.csv'
with open(new_csv_file, 'w', newline='') as new_csvfile:
    csv_writer = csv.writer(new_csvfile)
    csv_writer.writerow(['Prompt', 'Outline', 'Story'])
    for triplet in triplets:
        csv_writer.writerow(triplet)

print(f"New CSV file created: {new_csv_file}")