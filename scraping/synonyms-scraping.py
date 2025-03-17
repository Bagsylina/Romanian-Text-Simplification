import scrapy
import spacy
from bs4 import BeautifulSoup
import json
from os.path import join, dirname

NLP = spacy.load("ro_core_news_lg")

class SynonymsSpider(scrapy.Spider):
    name = 'synonyms'
    start_urls = ['https://www.dictionardesinonime.ro/']
    synonym_dictionary = {}
    antonym_dictionary = {}

    #parse pages for each letter
    def parse(self, response):
        letters_urls = response.css('div.content_page_simple a::attr(href)').getall()
        for url in letters_urls:
            yield scrapy.Request(url, callback=self.parse_letter)

    #parse pages for each word
    def parse_letter(self, response):
        words_urls = response.css('div.cont-list-words a::attr(href)').getall()
        for url in words_urls:
            yield scrapy.Request(url, callback=self.parse_word)

    #parse page for each word ang get its' synonyms
    def parse_word(self, response):
        word_list = response.css('span.def').getall()

        for word_data in word_list:
            #parse html
            soup = BeautifulSoup(word_data, 'html.parser')
            word = soup.span.b.text

            #if current set of words is of antonyms
            if '≠' in word_data:
                word = word.lower()
                word_nlp = NLP(word)
                word_lemma = ""

                # antonyms for verbs can have form at infinity
                if word_nlp[0].text == "a":
                    if word_nlp[1].text == "(":
                        word_lemma = word_nlp[4].lemma_
                    elif word_nlp[1].text == "se":
                        word_lemma = word_nlp[2].lemma_
                    else:
                        word_lemma = word_nlp[1].lemma_
                else:
                    word_lemma = word_nlp[0].lemma_

                #get all antonyms for word
                antonyms = soup.find_all('a')
                antonyms = [antonym.text for antonym in antonyms]
                antonyms_lemmas = [NLP(antonym)[0].lemma_ for antonym in antonyms]

                #add to antonym dictionary
                if word_lemma in self.antonym_dictionary:
                    self.antonym_dictionary[word_lemma].extend(antonyms_lemmas)
                else:
                    self.antonym_dictionary[word_lemma] = antonyms_lemmas

                continue

            #replace accent letters with original letter
            word = word.replace('Á', 'A')
            word = word.replace('É', 'E')
            word = word.replace('Í', 'I')
            word = word.replace('Ó', 'O')
            word = word.replace('Ú', 'U')

            #lowercase word
            word = word.lower()

            #get lemma for word
            word_lemma = NLP(word)[0].lemma_
            
            #get all synonyms for word
            synonyms = soup.find_all('a')
            synonyms = [synonym.text for synonym in synonyms]
            synonyms_lemmas = [NLP(synonym)[0].lemma_ for synonym in synonyms]

            #add to synonym dictionary
            if word_lemma in self.synonym_dictionary:
                self.synonym_dictionary[word_lemma].extend(synonyms_lemmas)
            else:
                self.synonym_dictionary[word_lemma] = synonyms_lemmas

    def closed(self, reason):
        with open(join(dirname(__file__), 'synonyms.json'), 'w') as f:
            json.dump(self.synonym_dictionary, f, indent=4)

        with open(join(dirname(__file__), 'antonyms.json'), 'w') as f:
            json.dump(self.antonym_dictionary, f, indent=4)