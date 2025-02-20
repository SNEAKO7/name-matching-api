# Farmer Name Matching API for Government Database

## Overview
This API service addresses name matching challenges in West Bengal's government farmer database by handling Bengali phonetic variations and transliteration inconsistencies. Built for the Department of Agriculture, it helps reconcile farmer records across different ID systems (Aadhaar, Bank, Kisan Credit Card) using advanced NLP techniques.

## Key Features
- **Phonetic Matching**: Uses Double Metaphone algorithm for Bengali-English transliterations
- **Hybrid Similarity Scoring**: Combines TF-IDF (25%) and Fuzzy Matching (75%)
- **SQL Integration**: Works with existing MySQL farmer registration database
- **Batch Processing**: Handles 10,000+ records per batch
- **Dockerized Deployment**: Containerized for easy cloud deployment

## Technology Stack
- **Backend**: FastAPI
- **Database**: MySQL (SQLAlchemy ORM)
- **ML/NLP**: scikit-learn, RapidFuzz, Metaphone
- **Infra**: Docker, Uvicorn
- **Monitoring**: Built-in logging (app.log)

API Endpoints
POST /update_farmer_registration - Processes registration table

POST /update_farmer_details - Processes details table

Configuration-
DATABASE_URL=mysql+pymysql://user:password@host:port/database
TFIDF_WEIGHT=0.25
FUZZY_WEIGHT=0.75
THRESHOLD=0.45



Post links for updating each table in database-
farmer_details-
http://127.0.0.1:8000/update_farmer_details?tfidf_weight=0.25&fuzzy_weight=0.75

farmer_registration-
http://127.0.0.1:8000/update_farmer_registration?tfidf_weight=0.25&fuzzy_weight=0.75


querry to select table and coloumns for both tables-
SELECT id, name_registration, name_aadhaar, name_kb, name_bank, 
       ai_aadhaar_name_match_flag, ai_aadhaar_name_match_accuracy, 
       ai_kb_name_match_flag, ai_kb_name_match_accuracy, 
       ai_bank_name_match_flag, ai_bank_name_match_accuracy
FROM farmer_registration;

SELECT id, name_registration, name_aadhaar, name_kb, name_bank, 
       ai_aadhaar_name_match_flag, ai_aadhaar_name_match_accuracy, 
       ai_kb_name_match_flag, ai_kb_name_match_accuracy, 
       ai_bank_name_match_flag, ai_bank_name_match_accuracy
FROM farmer_details;
