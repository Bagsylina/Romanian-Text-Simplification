import matplotlib.pyplot as plt
import json

with open('simplification_rating/model_comparisons.json', 'r') as f:
    models_results = json.load(f)