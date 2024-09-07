import pandas as pd


def create_statistical_terms_dict(file_path):
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(file_path, header=None, names=["Word"])

    # Create the statistical_terms dictionary
    statistical_terms = {}
    for i, word in enumerate(df["Word"]):
        if i != 0:
            term_key = f"term_{i}"
            term_value = [
                word,
                "",
            ]  # You can modify the second element of the list if needed
            statistical_terms[term_key] = term_value

    return statistical_terms


# Replace 'your_file_path.xlsx' with the actual path to your Excel file
file_path = "media/data/Book1.xlsx"
statistical_terms_dict = create_statistical_terms_dict(file_path)

# Print the resulting dictionary
print(statistical_terms_dict)
