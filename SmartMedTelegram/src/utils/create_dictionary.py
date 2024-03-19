import pandas as pd


def create_statistical_terms_dict(file_path):
    df = pd.read_excel(file_path, header=None, names=["Word"])

    statistical_terms = {}
    for i, word in enumerate(df["Word"]):
        if i != 0:
            term_key = f"term_{i}"
            term_value = [word, ""]
            statistical_terms[term_key] = term_value

    return statistical_terms


file_path = "media/data/Book1.xlsx"
statistical_terms_dict = create_statistical_terms_dict(file_path)

print(statistical_terms_dict)
