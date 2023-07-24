import pandas as pd
import os

def process_drug_class(row):
    drug_class = row["drug class"]
    drug_classes = drug_class.replace(",", " ").split("|")
    prompt = f"drug class is {' and '.join(drug_classes)}.generate smile image."
    return prompt

data_name = "LiverTox" # LiverTox|LactMed
csv_data = pd.read_csv(f"datasets/{data_name}.csv")

img_filenames = os.listdir(f"datasets/{data_name}")
filenames = [item.replace(".jpg", "") for item in img_filenames]

names = csv_data["drug name"].tolist()

real_names = set(names) & set(filenames)

csv_data = csv_data[csv_data["drug name"].isin(real_names)]
csv_data["drug class"] = csv_data.apply(process_drug_class, axis=1)

csv_data.to_csv(f"datasets/{data_name}_data.csv", index=False)