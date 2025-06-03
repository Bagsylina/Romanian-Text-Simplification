from transformers import pipeline
import spacy
import fasttext.util
from wordfreq import zipf_frequency
from sklearn.preprocessing import MinMaxScaler
import json
import scipy.spatial, scipy.special

fasttext.util.download_model('ro', if_exists='ignore')

class RoTextSimpModel:
    NLP = spacy.load("ro_core_news_lg")
    ft = fasttext.load_model('cc.ro.300.bin')


    def __init__(self, masked_model=pipeline('fill-mask', model='dumitrescustefan/bert-base-romanian-cased-v1', top_k=10)):
        self.unmasker = masked_model
        
        #load synonyms and antonyms
        with open('scraping/synonyms.json', 'r') as f:
            self.synonyms_dict = json.load(f)

        with open('scraping/antonyms.json', 'r') as f:
            self.antonyms_dict = json.load(f)

        #load dex online data
        with open('dex-online-database/word_inflections.json', 'r') as f:
            self.word_inflections = json.load(f)

        with open('dex-online-database/spaCy_tags.json', 'r') as f:
            self.spacy_tags = json.load(f)

        with open('dex-online-database/pron_and_num_pairs.json', 'r', encoding='utf-8') as f:
            self.pron_pairs = json.load(f)


    #preproccesing sentence to replace diacritics in a single format
    def preprocess_sentence(self, sentence):
        #get right versions of ș/ț
        sentence = sentence.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")

        #transform a/i + unicode 770/u+0302 in â/î
        sentence = sentence.replace("a\u0302", "â").replace("i\u0302", "î").replace("A\u0302", "Â").replace("I\u0302", "Î")

        #transform s/t + unicode 807/u+0327 or 806/u+0326 in ș/ț
        sentence = sentence.replace("s\u0327", "ș").replace("t\u0327", "ț").replace("S\u0327", "Ș").replace("T\u0327", "Ț")
        sentence = sentence.replace("s\u0326", "ș").replace("t\u0326", "ț").replace("S\u0326", "Ș").replace("T\u0326", "Ț")

        #transform a + unicode 774/u+0306 in ă
        sentence = sentence.replace("a\u0306", "ă").replace("A\u0307", "Ă")

        return sentence
    
    
    #check if token is valid to be replaced by checking multiple features and its' zipf score
    def valid_token(self, token, zipf_score):
        #if token is punctuation
        if token.is_punct:
            return False
        
        #if token is number
        if token.is_digit or token.like_num:
            return False
        
        #if token is whitespace
        if token.is_space:
            return False
        
        #if token is currency:
        if token.is_currency:
            return False
        
        #if token is url or email
        if token.like_url or token.like_email:
            return False
        
        #should be replaced only tokens that are nouns, adjectives, adverbs or verbs (proper nouns are also excluded here)
        if token.pos_ not in ["NOUN", "ADJ", "ADV", "VERB", "AUX"]:
            return False
        
        #if token is proper noun
        if token.pos_ == "NOUN" and token.tag_[1] == 'p':
            return False
        
        #exclude capitalised words if not at the start of a sentence (it's a name)
        if not token.is_sent_start and token.text[0].isupper():
            return False

        #exclude cardinal directions
        cardinal_directions = ["nord", "est", "sud", "vest", "nord-est", "nord-vest", "sud-est", "sud-vest", "nordic", "estic", "sudic", "vestic", "nord-estic", "nord-vestic", "sud-estic", "sud-vestic"]
        if token.lemma_ in cardinal_directions:
            return False

        #exclude stop words (should not be replaced as they are frequently used words)
        if token.is_stop:
            return False
        
        #too high zipf score
        if zipf_score > 4.25:
        #if zipf_score > 4.5:
        #if zipf_score > 4:
            return False
        
        return True
    

    #in a sentence find words that can be simplified
    def word_candidates(self, sentence, ignore_list):
        tokens = self.NLP(sentence)
        replacable_tokens = []

        #sort tokens by Zipf score
        for token in tokens:
            token_score = zipf_frequency(token.text, 'ro')
            #token_score = zipf_frequency(token.lemma_, "ro")
            if self.valid_token(token, token_score) and token.text not in ignore_list:
                replacable_tokens.append((token, token_score))

        replacable_tokens.sort(key=lambda x: x[1])

        return replacable_tokens
    

    #for nouns, get a sentence if that noun was the opposite gender
    def noun_gender_change(self, sentence, token):
        changed_tokens_list = []
        sentence_tokens = self.NLP(sentence)

        #get related words (adjectives, numerals, pronouns) related to the word to be replaced
        token_children = [child.i for child in token.children]
        token_ancestors = [ancestor.i for ancestor in token.ancestors]

        #get related words (adjectives, numerals, pronouns) related to the word to be replaced
        token_children = [child.i for child in token.children]
        token_ancestors = [ancestor.i for ancestor in token.ancestors]

        #change gender for related words in the sentence
        for sent_token in sentence_tokens:
            changed = False
            if sent_token.i in token_children or sent_token.i in token_ancestors:
                #if word is adjective search if opposite gender is available
                if sent_token.pos_ == "ADJ":
                    if sent_token.tag_ in self.spacy_tags:
                        tag_id = self.spacy_tags[sent_token.tag_]["dex_online_id"][0]
                        if sent_token.tag_[3] == "m":
                            tag_id += 8
                        else:
                            tag_id -= 8
                        tag_id = str(tag_id)

                        #change gender for adjective if form is found
                        if sent_token.lemma_ in self.word_inflections:
                            if tag_id in self.word_inflections[sent_token.lemma_]["inflections"]:
                                changed_tokens_list.append(self.word_inflections[sent_token.lemma_]["inflections"][tag_id])
                                changed = True
                
                #if in list get opposite gender
                elif sent_token.text in self.pron_pairs:
                    changed_tokens_list.append(self.pron_pairs[sent_token.text])
                    changed = True

                    #replace article if present
                    for child in sent_token.children:
                        if child.text == "a":
                            changed_tokens_list[child.i] = "al"
                        elif child.text == "al":
                            changed_tokens_list[child.i] = "a"
                        elif child.text == "ai":
                            changed_tokens_list[child.i] = "ale"
                        elif child.text == "ale":
                            changed_tokens_list[child.i] = "ai"

                #if ordinal numeral change gender
                elif sent_token.pos_ == "NUM" and sent_token.tag_[1] == "o" and sent_token.lemma != "primul":
                    tag_id = 45
                    if sent_token.tag_[2] == "f":
                        tag_id = 41
                    tag_id = str(tag_id)
                    sent_token_lemma = sent_token.lemma_ + "lea"
                    if sent_token_lemma in self.word_inflections:
                        changed_tokens_list.append(self.word_inflections[sent_token_lemma]["inflections"][tag_id])
                    else:
                        changed_tokens_list.append(sent_token_lemma)
                    changed = True

            #mask wanted token
            elif sent_token.i == token.i:
                changed_tokens_list.append("[MASK]")
                changed = True

            if not changed:
                changed_tokens_list.append(sent_token.text)

        #build sentence with changed genders and get suggestions
        changed_sentence = ""
        for sent_token in changed_tokens_list:
            if sent_token in ",.;!?:-_)]}":
                changed_sentence = changed_sentence.rstrip()
                changed_sentence += sent_token + " "
            elif sent_token[:-1] == "-" or sent_token in "([{":
                changed_sentence += sent_token
            else:
                changed_sentence += sent_token + " "

        return changed_sentence
    

    #generate possible replacements for a word in a sentence
    def substitution_generation(self, sentence, token):
        # generate a masked sentence in this form: [CLS] original sentence [SEP] masked sentence [SEP]
        masked_sentence = sentence[:token.idx] + "[MASK]" + sentence[token.idx + len(token.text):]
        model_sentence = "[CLS] " + sentence + " [SEP] " + masked_sentence + " [SEP]"
        result = self.unmasker(model_sentence)
        suggestions = []

        for x in result:
            #check if suggestion is same part of speech 
            replaced_sentence = sentence[:token.idx] + x["token_str"] + sentence[token.idx + len(token.text):]
            replaced_tokens = self.NLP(replaced_sentence)
            suggestion_token = replaced_tokens[token.i]

            if token.pos != suggestion_token.pos and not (token.pos_ == "AUX" and suggestion_token.pos_ == "VERB") and not (token.pos_ == "VERB" and suggestion_token.pos_ == "AUX"):
                continue

            #incomplete word
            if x["token_str"][0] == "#":
                continue

            x["changed"] = False
            suggestions.append(x)

        return suggestions
    

    #special case of suggestions for nouns
    def noun_substitution_generation(self, sentence, changed_sentence, token):
        #get original suggestions
        suggestions = self.substitution_generation(sentence, token)

        #generate a masked sentence in this form: [CLS] original sentence [SEP] masked sentence [SEP] for the changed_sentence as well
        changed_model_sentence = "[CLS] " + sentence + " [SEP] " + changed_sentence + " [SEP]"
        changed_result = self.unmasker(changed_model_sentence)

        #filter suggestions for changed sentence and append them to the result list
        for x in changed_result:
            #check if suggestion is same part of speech 
            replaced_sentence = changed_sentence.replace("[MASK]", x["token_str"])
            replaced_tokens = self.NLP(replaced_sentence)
            
            if token.i < len(replaced_tokens):
                suggestion_token = replaced_tokens[token.i]

                if token.pos != suggestion_token.pos and not (token.pos_ == "AUX" and suggestion_token.pos_ == "VERB") and not (token.pos_ == "VERB" and suggestion_token.pos_ == "AUX"):
                    continue

                #incomplete word
                if x["token_str"][0] == "#":
                    continue

                if len(token.tag_) > 2 and len(suggestion_token.tag_) > 2 and token.tag_[2] != suggestion_token.tag_[2]:
                    x["changed"] = True
                else:
                    x["changed"] = False
                suggestions.append(x)

        return suggestions
    

    #additional score for ranking, based of if replacement and original word and synonyms or antonyms
    def synonym_score(self, suggestion, token):
        sugg_token = self.NLP(suggestion)[0]
        sugg_lemma = sugg_token.lemma_
        token_lemma = token.lemma_

        #positive score if synonyms
        if (token_lemma in self.synonyms_dict and sugg_lemma in self.synonyms_dict[token_lemma]) or (sugg_lemma in self.synonyms_dict and token_lemma in self.synonyms_dict[sugg_lemma]):
            return 1

        #negative score if antonyms
        if (token_lemma in self.antonyms_dict and sugg_lemma in self.antonyms_dict[token_lemma]) or (sugg_lemma in self.antonyms_dict and token_lemma in self.antonyms_dict[sugg_lemma]):
            return 0
        if "ne" + token_lemma == sugg_lemma or "ne" + sugg_lemma == token_lemma:
            return 0
        if "in" + token_lemma == sugg_lemma or "in" + sugg_lemma == token_lemma:
            return 0
        
        #negative score if same word to encourage different suggestions
        if token_lemma == sugg_lemma:
            return 0.25
        
        #netural score if not related
        return 0.5
    

    #rank given substitutions for a word based on multiple criteria
    def substitution_ranking(self, suggetions, token):
        substitutions = []

        bert_scores = []
        zipf_scores = []
        fasttext_scores = []
        synonym_scores = []

        #calculate a score for each suggestion based on BERT, Zipf, FastText and if words are synonyms
        for x in suggetions:
            bert_scores.append([x["score"]])

            zipf_score = zipf_frequency(x["token_str"], 'ro')
            zipf_scores.append([zipf_score])

            fasttext_score = 1 - scipy.spatial.distance.cosine(self.ft[token.text], self.ft[x["token_str"]])
            fasttext_scores.append([fasttext_score])

            syn_score = self.synonym_score(x["token_str"], token)
            synonym_scores.append([syn_score])
        
        #use min max scaler to normalize each set of the scores
        if len(bert_scores) > 0:
            bert_scaler = MinMaxScaler()
            zipf_scaler = MinMaxScaler()
            fasttext_scaler = MinMaxScaler()
            synonym_scaler = MinMaxScaler()

            bert_scores_norm = bert_scaler.fit_transform(bert_scores)
            zipf_scores_norm = zipf_scaler.fit_transform(zipf_scores)
            fasttext_scores_norm = fasttext_scaler.fit_transform(fasttext_scores)
            synonym_scores_norm = synonym_scaler.fit_transform(synonym_scores)

            #final score is average of all 4
            for i in range(len(suggetions)):
                final_score = (bert_scores_norm[i] + zipf_scores_norm[i] + fasttext_scores_norm[i] + synonym_scores_norm[i]) / 4
                substitutions.append((suggetions[i]["token_str"], suggetions[i]["changed"], final_score))

            #sort suggestions by score and replace the word with the best suggestion if it has a higher Zipf score
            substitutions.sort(reverse=True, key=lambda x: x[2])

        return substitutions
    

    #align replacement and original word to have the same form (time, case, gender, plurality) if possible
    def align_word_form(self, token, suggestion):
        #search if original word has inflections available for its' pos tag
        has_inflections = False
        inflection_ids = []

        #special case for participe verbs, take form after adjective
        if token.tag_[:3] == "Vmp":
            if token.lemma_ in self.word_inflections:
                if token.tag_ in self.spacy_tags:
                    has_inflections = True
                    inflection_ids = self.spacy_tags[token.tag_]["adj_id"]

            #take lemma as the one of the adjective
            suggestion_lemma = self.NLP(suggestion)[0].lemma_
            if suggestion_lemma in self.word_inflections:
                if "52" in self.word_inflections[suggestion_lemma]["inflections"]:
                    suggestion_lemma = self.word_inflections[suggestion_lemma]["inflections"]["52"]
                
            #check if replacement also has the same forms available
            if has_inflections and suggestion_lemma in self.word_inflections:
                for id in inflection_ids:
                    str_id = str(id)
                    if str_id in self.word_inflections[suggestion_lemma]["inflections"]:
                        if token.is_title:
                            return self.word_inflections[suggestion_lemma]["inflections"][str_id].title()
                        return self.word_inflections[suggestion_lemma]["inflections"][str_id]
            
            if token.is_title:
                return suggestion.title()
            return suggestion

        #get inflection form ids for original word
        if token.lemma_ in self.word_inflections:
            if token.tag_ in self.spacy_tags:
                has_inflections = True
                inflection_ids = self.spacy_tags[token.tag_]["dex_online_id"]

        #replace the top replacement with the right form if found
        suggestion_lemma = self.NLP(suggestion)[0].lemma_

        #check if replacement also has the same forms available
        if has_inflections and suggestion_lemma in self.word_inflections:
            for id in inflection_ids:
                str_id = str(id)
                if str_id in self.word_inflections[suggestion_lemma]["inflections"]:
                    if token.is_title:
                        return self.word_inflections[suggestion_lemma]["inflections"][str_id].title()
                    return self.word_inflections[suggestion_lemma]["inflections"][str_id]

        #if not replacement is found return original word
        if token.is_title:
            return suggestion.title()
        return suggestion
    

    #simplify a sentence
    def sentence_simplification(self, sentence):
        token_replaced = True
        #set of tokens that were already replaced in the sentence
        ignore_list = set([])

        #preprocessing
        sentence = self.preprocess_sentence(sentence)

        while token_replaced:
            token_replaced = False

            #get list of tokens to be replaced
            replacable_tokens = self.word_candidates(sentence, ignore_list)
            
            for token in replacable_tokens:
                if token_replaced:
                    break

                token_text = token[0].text
                changed_sentence = sentence

                #different suggestions for nouns
                if token[0].pos_ == "NOUN":
                    changed_sentence = self.noun_gender_change(sentence, token[0])
                    suggestions = self.noun_substitution_generation(sentence, changed_sentence, token[0])
                else:
                    suggestions = self.substitution_generation(sentence, token[0])
                    
                suggestions = self.substitution_ranking(suggestions, token[0])

                if len(suggestions) == 0:
                    continue
                
                #take top replacement and replace the word (if less complex)
                top_replacement = suggestions[0][0]

                if zipf_frequency(top_replacement, 'ro') >= zipf_frequency(token_text, 'ro'):
                    top_replacement = self.align_word_form(token[0], top_replacement)
                    if suggestions[0][1] == False:
                        sentence = sentence[:token[0].idx] + top_replacement + sentence[token[0].idx + len(token_text):]
                    else:
                        sentence = changed_sentence.replace("[MASK]", top_replacement) 
                    ignore_list.add(top_replacement)
                    token_replaced = True

        return sentence
    

    #simplify a text by splitting it in multiple sentences
    def text_simplification(self, text):
        doc = self.NLP(text)
        for sentence in doc.sents:
            new_sentence = self.sentence_simplification(sentence.text)
            text = text.replace(sentence.text, new_sentence)
        return text
    

    #get simplification suggestions for a certain word in a text
    def word_simplifications(self, text, word):
        #preprocessing both text and word
        text = self.preprocess_sentence(text)
        word = self.preprocess_sentence(word)

        doc = self.NLP(text)
        word_lemma = self.NLP(word)[0].lemma_
        simplification_suggestions = []

        #go through each sentence and if word is found generate simplifications
        sent_index = 0
        for sentence in doc.sents:
            sentence_tokens = self.NLP(sentence.text)

            for token in sentence_tokens:
                if token.text == word or token.lemma_ == word_lemma:
                    suggestions = []

                    if token.pos_ == "NOUN":
                        changed_sentence = self.noun_gender_change(sentence.text, token)
                        suggestions = self.noun_substitution_generation(sentence.text, changed_sentence, token)
                    else:
                        suggestions = self.substitution_generation(sentence.text, token)

                    suggestions = self.substitution_ranking(suggestions, token)

                    simplification_suggestions.append({"word": token.text, "sent_index": sent_index, "word_index": token.i, "suggestions": suggestions})

            sent_index += 1

        return simplification_suggestions
    

"""
unmasker = pipeline('fill-mask', model='dumitrescustefan/bert-base-romanian-cased-v1', top_k=10)
simpModel = RoTextSimpModel(unmasker)
text = "În informatică, un procesor este un dispozitiv hardware al unui computer care pornind de la un set de instrucțiuni efectuează operațiuni pe o sursă externă de date. Termenul este frecvent utilizat pentru a face referire la unitatea centrală de procesare dintr-un sistem. Procesorul este elementul principal al unui sistem de calcul și încorporează funcțiile unității centrale de prelucrare a informației a unui calculator sau a unui sistem electronic structurat funcțional (care coordonează sistemul). De obicei, fizic procesorul se prezintă sub forma unui microprocesor, care este fabricat pe un singur cip de circuit integrat metal-oxid-semiconductor (MOS). Reprezintă forma structurală cea mai complexă pe care o pot avea circuitele integrate. Cipul semiconductor, care este plasat pe placa de bază, este foarte complex, putând ajunge să conțină milioane de microtranzistoare. El controlează activitățile întregului sistem în care este integrat și poate prelucra datele furnizate de utilizator. Procesorul asigură procesarea instrucțiunilor și datelor, atât din sistemul de operare al sistemului, cât și din aplicațiile utilizatorului, și anume le interpretează, prelucrează și controlează, execută sau supervizează transferurile de informații și controlează activitatea generală a celorlalte componente care alcătuiesc un sistem de calcul."
print(simpModel.text_simplification(text))
print(simpModel.word_simplifications(text, "procesor"))

unmaskerMulti = pipeline('fill-mask', model='google-bert/bert-base-multilingual-cased', top_k=10)
simpModelMulti = RoTextSimpModel(unmaskerMulti)
print(simpModelMulti.text_simplification(text))
"""

"""
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gpt4omlm import get_substitution_suggestions

unmasker = get_substitution_suggestions
simpModel = RoTextSimpModel(unmasker)
text = "În informatică, un procesor este un dispozitiv hardware al unui computer care pornind de la un set de instrucțiuni efectuează operațiuni pe o sursă externă de date. Termenul este frecvent utilizat pentru a face referire la unitatea centrală de procesare dintr-un sistem. Procesorul este elementul principal al unui sistem de calcul și încorporează funcțiile unității centrale de prelucrare a informației a unui calculator sau a unui sistem electronic structurat funcțional (care coordonează sistemul). De obicei, fizic procesorul se prezintă sub forma unui microprocesor, care este fabricat pe un singur cip de circuit integrat metal-oxid-semiconductor (MOS). Reprezintă forma structurală cea mai complexă pe care o pot avea circuitele integrate. Cipul semiconductor, care este plasat pe placa de bază, este foarte complex, putând ajunge să conțină milioane de microtranzistoare. El controlează activitățile întregului sistem în care este integrat și poate prelucra datele furnizate de utilizator. Procesorul asigură procesarea instrucțiunilor și datelor, atât din sistemul de operare al sistemului, cât și din aplicațiile utilizatorului, și anume le interpretează, prelucrează și controlează, execută sau supervizează transferurile de informații și controlează activitatea generală a celorlalte componente care alcătuiesc un sistem de calcul."
print(simpModel.text_simplification(text))
print(simpModel.word_simplifications(text, "procesor"))
"""