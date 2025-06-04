import pandas as pd
import numpy as np

data= pd.read_csv('stock(updated).csv')
#print(data.info())

irrelevant_categories = [
    'COMEDY', 'PARENTING', 'CULTURE & ARTS', 'ENTERTAINMENT','WEIRD NEWS', 'STYLE & BEAUTY', 'HOME & LIVING','SPORTS','FOOD & DRINK','TRAVEL','CRIME','WELLNESS','ENVIRONMENT','MEDIA','WOMEN','SCIENCE','TECH','BLACK VOICES','QUEER VOICES','EDUCATION']
data = data[~data['category'].isin(irrelevant_categories)]
print(data['category'].unique())
print(f"Shape of the DataFrame: {data.shape}") 
data.to_csv("stock(updated).csv", index=False)
