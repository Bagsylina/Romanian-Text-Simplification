import mysql.connector
import json
from os.path import join, dirname

conn = mysql.connector.connect(
    host="localhost",
    port="3306",
    user="root",
    password="root",
    database="dexonline"
)

cursor = conn.cursor(dictionary=True)

cursor.execute("SELECT id, formUtf8General FROM lexeme;")
words = cursor.fetchall();

inflection_dict = {}

for word in words:
    cursor.execute("SELECT inflectionId, formUtf8General, apheresis FROM inflectedform WHERE lexemeId = {id};".format(id = word["id"]))
    inflections = cursor.fetchall()
    text = word["formUtf8General"]

    if len(inflections) > 1:
        if text not in inflection_dict:
            inflection_dict[text] = {"id": word["id"], "word": text, "inflections": {}}

            for inflection in inflections:
                if inflection["apheresis"] == 0:
                    inflection_dict[text]["inflections"][inflection["inflectionId"]] = inflection["formUtf8General"]
        
        else:
            for inflection in inflections:
                if inflection["apheresis"] == 0 and inflection["inflectionId"] not in inflection_dict[text]["inflections"]:
                    inflection_dict[text]["inflections"][inflection["inflectionId"]] = inflection["formUtf8General"]

with open(join(dirname(__file__), 'word_inflections.json'), 'w') as f:
    json.dump(inflection_dict, f)

cursor.close()
conn.close()