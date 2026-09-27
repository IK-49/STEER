import numpy as np
import pandas as pd


df = pd.read_csv("pre_cleaned_data.csv", dtype=str)

# Rename column names to snake case
df = df.rename(columns={"Name": "School_Name", "FIPS County Code": "FIPS_County_Code", "Composite Score": "Composite_Score", "County": "District_Name"})

# Resolve duplicate school names
df["Display_Name"] = (
    df["School_Name"].fillna("School Name Unavailable")
    + " ("
    + df["City"].fillna("N/A")
    + ", "
    + df["State"].fillna("N/A")
    + ") - NCES"
    + df["NCESSCH"].fillna("ID Unavailable")
)

# Drop duplicate data points
df = df.drop_duplicates()

print(df.head(5))