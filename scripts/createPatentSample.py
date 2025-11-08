#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np 
import os
import zipfile 
import matplotlib.pyplot as plt
from tqdm import tqdm

# Patent Folder
patentFolder = "/project/jevans/apto_data_engineering/data/PatentsView/Granted Patents/Main"

cpc_classes = {
    "G": {
        "title": "Physics",
        "subclasses": {
            "G06": "Computing; Calculating; Counting",
            "G06C": "Digital Computers in General",
            "G06F": "Electric Digital Data Processing",
            "G06K": "Recognition of Data; Presentation of Data; Record Carriers; Handling Record Carriers",
            "G06N": "Computer Systems Based on Specific Computational Models (e.g., neural networks, quantum computing)",
            "G06Q": "Data Processing Systems or Methods, Specially Adapted for Administrative, Commercial, Financial, Managerial, Supervisory, or Forecasting Purposes",
            "G11": "Information Storage",
            "G11B": "Information Storage Based on Relative Movement Between Record Carrier and Transducer",
            "G11C": "Static Stores",
        }
    },
    "H": {
        "title": "Electricity",
        "subclasses": {
            "H03": "Basic Electronic Circuitry",
            "H03M": "Coding, Decoding, or Code Conversion, in General",
            "H04": "Electric Communication Technique",
            "H04L": "Transmission of Digital Information",
            "H04W": "Wireless Communication Networks",
            "H05": "Electric Techniques Not Otherwise Provided For",
            "H05B": "Electric Heating; Electric Lighting Not Otherwise Provided For",
            "H05K": "Printed Circuits; Casings or Constructional Details of Electric Apparatus",
        }
    }
}

print("Loading patents data...")
zip_file_path = patentFolder + '/g_patent.tsv.zip'
tsv_file_name = 'g_patent.tsv'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    with zip_ref.open(tsv_file_name) as tsv_file:
        patents_df = pd.read_csv(tsv_file, sep='\t',low_memory=False)

# remove them
patents_df = patents_df[~patents_df['patent_id'].str.match(r'^[^0-9]', na=False)]
# convert to numeric
patents_df['patent_id'] = pd.to_numeric(patents_df['patent_id'])

patents_df = patents_df[patents_df['patent_type']=='utility'].copy()
patents_df.drop(columns=['patent_type','filename','withdrawn','num_claims','wipo_kind'],inplace=True)

patents_df['patentGrant_year'] = pd.to_datetime(patents_df['patent_date']).dt.year
patents_df['patentGrant_month'] = pd.to_datetime(patents_df['patent_date']).dt.month
patents_df['patentGrant_year_month'] = pd.to_datetime(patents_df['patent_date']).dt.to_period('M')

print("Loading CPC titles...")
zip_file_path = patentFolder + '/g_cpc_title.tsv.zip'
tsv_file_name = 'g_cpc_title.tsv'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    with zip_ref.open(tsv_file_name) as tsv_file:
        cpcTitles = pd.read_csv(tsv_file, sep='\t',low_memory=False)

cpcTitles = cpcTitles[['cpc_class','cpc_class_title','cpc_subclass','cpc_subclass_title']].drop_duplicates()

print("Loading CPC classifications...")
clean_cpcPath = patentFolder + "/clean_cpc_class.csv"
if os.path.exists(clean_cpcPath):
    cpc_class = pd.read_csv(clean_cpcPath)
    print("Loaded from saved file")
else:
    print('Constructing Data')
    zip_file_path = '/project/jevans/nadav/impulseResponse/patentData/g_cpc_current.tsv.zip'
    tsv_file_name = 'g_cpc_current.tsv'
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        with zip_ref.open(tsv_file_name) as tsv_file:
            cpc_class = pd.read_csv(tsv_file, sep='\t')
    
    # removes patents ids that start with non-numeric characters
    cpc_class = cpc_class[~cpc_class['patent_id'].astype(str).str.match(r'^[^0-9]', na=False)]
    cpc_class['patent_id'] = pd.to_numeric(cpc_class['patent_id'])
    cpc_class = cpc_class[cpc_class['cpc_type'] == 'inventional']
    
    # restriction Only to classes 
    cpc_class.drop(columns=['cpc_group','cpc_type'],inplace=True)
    cpc_class.drop_duplicates(inplace=True)

    cpc_class.sort_values(['patent_id','cpc_sequence'],inplace=True )
    cpc_class = cpc_class.merge(cpcTitles, on = ["cpc_class",'cpc_subclass'],how='inner',validate='m:1')

print("Merging patents with CPC classifications...")
patentsCPC = pd.merge(patents_df, cpc_class, on='patent_id',how='left')
print(patentsCPC.columns)

print("Loading patent abstracts...")
zip_file_path = patentFolder + '/g_patent_abstract.tsv.zip'
tsv_file_name = 'g_patent_abstract.tsv'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    with zip_ref.open(tsv_file_name) as tsv_file:
        patentsAbstracts_df = pd.read_csv(tsv_file, sep='\t',low_memory=False)

# remove them
patentsAbstracts_df = patentsAbstracts_df[~patentsAbstracts_df['patent_id'].str.match(r'^[^0-9]', na=False)]
# convert to numeric
patentsAbstracts_df['patent_id'] = pd.to_numeric(patentsAbstracts_df['patent_id'])

print("Merging abstracts...")
patentsCPC = patentsCPC.merge(patentsAbstracts_df,on='patent_id',how='left',validate="m:1")
print(patentsCPC.head(5))

print("Creating computer science patent indicator...")
valid_classes = set()
for section in cpc_classes.values():
    valid_classes.update(section["subclasses"].keys())

# Create an indicator column
patentsCPC["is_computer_science_patent"] = (patentsCPC["cpc_subclass"].isin(valid_classes)) | (patentsCPC["cpc_class"].isin(valid_classes)) 

print("Loading AI patents data...")
zip_file_path = "/project/jevans/apto_data_engineering/data/PatentsView/AIPatents/ai_model_predictions.csv.zip"
tsv_file_name = 'ai_model_predictions.csv'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    with zip_ref.open(tsv_file_name) as tsv_file:
        ai_patents = pd.read_csv(tsv_file,low_memory=False)

# convert to numeric
ai_patents = ai_patents[ai_patents['flag_patent']==1]
ai_patents = ai_patents[~ai_patents['doc_id'].str.match(r'^[^0-9]', na=False)]
ai_patents['doc_id'] = pd.to_numeric(ai_patents['doc_id'])

print("Merging AI patents...")
patentsCPC = patentsCPC.merge(ai_patents,left_on='patent_id',right_on='doc_id',how='left',validate="m:1")

print("Loading brief summary texts...")
dfs = []
for year in tqdm(range(1976, 2025),desc='year gone by'): 
    
    filename = f"g_brf_sum_text_{year}.tsv.zip" 
    tsv_file_name = f"g_brf_sum_text_{year}.tsv" 
    zip_file_path = "/project/jevans/apto_data_engineering/data/PatentsView/Granted Patents2/Brief Summary Text/" + filename
    
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            with zip_ref.open(tsv_file_name) as tsv_file:
                df = pd.read_csv(tsv_file, sep='\t', low_memory=False)
                dfs.append(df)  # Append DataFrame to list
    except Exception as e:
        print(f"Error processing {year}: {e}")

print('done loading doing cleaning')
# Concatenate all DataFrames
patentsBreifSummary_df = pd.concat(dfs, ignore_index=True)
patentsBreifSummary_df = patentsBreifSummary_df[~patentsBreifSummary_df['patent_id'].astype(str).str.match(r'^[^0-9]', na=False)]
patentsBreifSummary_df['patent_id'] = pd.to_numeric(patentsBreifSummary_df['patent_id'])

print("Final merge with abstracts...")
patentsCPC = patentsCPC.merge(patentsAbstracts_df,on='patent_id',how='left',validate="m:1")

print("Saving sample of 10 patents...")
patentsCPC.to_parquet('/project/jevans/QuestionAnswerApproach/data/patentsData.parquet')
print(patentsCPC.head(5))

print("Sample Patent Classes")

patentsCPC = pd.read_parquet('/project/jevans/QuestionAnswerApproach/data/patentsData.parquet')
print(patentsCPC.head(3))

# Load your dataset
df = patentsCPC

# Make sure output directory exists
output_dir = "/project/jevans/QuestionAnswerApproach/data/patent_sample"
os.makedirs(output_dir, exist_ok=True)

# Group by cpc_class
for cpc_class, group in tqdm(df.groupby("cpc_class")):
    # Sample up to 10,000 entries (fewer if group is smaller)
    sample_df = group.sample(n=min(10000, len(group)), random_state=42)

    # Determine file prefix
    is_cs = sample_df["is_computer_science_patent"].mean() == 1
    if is_cs.all():
        print(cpc_class)
        prefix = "cs_class_"
    else:
        prefix = "class_"

    # Save to Parquet
    filename = f"{prefix}{cpc_class}.parquet"
    filepath = os.path.join(output_dir, filename)
    sample_df.to_parquet(filepath, index=False)

    print(f"Saved {len(sample_df)} rows to {filepath}")

print("Done!")




