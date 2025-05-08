from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)

gpt_prompt = 'Context: "{context}"\nQuestion: Given the above context, list ten alternative romanian one word substitutions for [MASK], each with a score. List only the words without translations, transcriptions or explanations, in the following format: "x. word - score".\nAnswer:'

def gpt4o_substitution_suggestions(context):
    gpt_input = gpt_prompt.format(context = context)
    suggestions = []
    successful = False
    nr_unsuccessful = 0

    while not successful:
        suggestions = []
        successful = True
        
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=gpt_input
            )

            response_text = response.output_text
            response_list = response_text.split('\n')

            for x in response_list: 
                response_split = x.split(' ')
                
                if response_split[1] == "se" or response_split[1] == "în":
                    suggested_word = response_split[2]
                    suggestion_score = float(response_split[4])
                else:
                    suggested_word = response_split[1]
                    suggestion_score = float(response_split[3])

                suggestions.append({'score': suggestion_score, 'token_str': suggested_word})

        except:
            successful = False
            nr_unsuccessful += 1

            if nr_unsuccessful == 10:
                return []

    return suggestions