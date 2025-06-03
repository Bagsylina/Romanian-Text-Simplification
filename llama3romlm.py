import requests
from model import RoTextSimpModel

openai_api_base = "http://gpu.hypha.ro:10101/v1/chat/completions"
model = "OpenLLM-Ro/RoLlama3-8b-Instruct-2025-04-23"

llama_prompt = 'Context: "{context}"\nDându-se următorul context, returnează o listă de zece cuvinte drept substituție pentru [MASK], fiecare cu câte un scor. Listează doar cuvintele fără explicații, în următorul format: "x. word - score".\nRăspuns:'

def llama3ro_substitution_suggestions(context):
    llama_input = llama_prompt.format(context = context)
    chat = [
        {
            "role": "user",
            "content": llama_input
        }
    ]
    json_data = {"model": model, "messages": chat}
    suggestions = []
    successful = False
    nr_unsuccessful = 0

    while not successful:
        suggestions = []
        successful = True
        
        try:
            response = requests.post(openai_api_base, json=json_data)

            response_text = response.json()['choices'][0]['message']['content']
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