import numpy as np
import pandas as pd


df = pd.read_csv("pre_cleaned_data.csv", dtype="string", na_values=["N/A", "NA", "", " "])

# Renames column names to snake case
df = df.rename(columns={"Name": "School_Name", "FIPS County Code": "FIPS_County_Code", "Composite Score": "Composite_Score", "County": "County_Name"})

# Removes leading and trailing whitespace
columns = ["NCESSCH", "School_Name", "FIPS_County_Code", "County_Name", "City", "Economic", "Education", "Health", "Housing", "Crime", "Composite_Score"]
for column in columns:
    df[column] = df[column].astype("string").str.strip()

score_columns = ["Economic", "Education", "Health", "Housing", "Crime", "Composite_Score"]

for column in score_columns:
    df[column] = pd.to_numeric(df[column],errors="coerce")

# Resolves duplicate school names
df["Display_Name"] = (df["School_Name"].fillna("School Name Unavailable")
    + " ("
    + df["City"].fillna("N/A")
    + ", "
    + df["State"].fillna("N/A")
    + ") - NCES "
    + df["NCESSCH"].fillna("ID Unavailable"))

# Drops duplicate data points
df = df.drop_duplicates()

df.to_csv("cleaned_data.csv", index=False)