#updated code for more accuracy with bengali advanced normalization



import logging
import re
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from metaphone import doublemetaphone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = "mysql+pymysql://dbadmin:dbuser%402023@172.25.144.165:3306/paddy_live"
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Farmer model
class Farmer(Base):
    __tablename__ = "farmer_registration"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name_registration = Column(String(255))
    name_aadhaar = Column(String(255))
    name_kb = Column(String(255))
    name_bank = Column(String(255))
    ai_aadhaar_name_match_flag = Column(Boolean, default=False)
    ai_aadhaar_name_match_accuracy = Column(Float, default=0.0)
    ai_kb_name_match_flag = Column(Boolean, default=False)
    ai_kb_name_match_accuracy = Column(Float, default=0.0)
    ai_bank_name_match_flag = Column(Boolean, default=False)
    ai_bank_name_match_accuracy = Column(Float, default=0.0)

# FarmerDetails model
class FarmerDetails(Base):
    __tablename__ = "farmer_details"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name_registration = Column(String(255))
    name_aadhaar = Column(String(255))
    name_kb = Column(String(255))
    name_bank = Column(String(255))
    ai_aadhaar_name_match_flag = Column(Boolean, default=False)
    ai_aadhaar_name_match_accuracy = Column(Float, default=0.0)
    ai_kb_name_match_flag = Column(Boolean, default=False)
    ai_kb_name_match_accuracy = Column(Float, default=0.0)
    ai_bank_name_match_flag = Column(Boolean, default=False)
    ai_bank_name_match_accuracy = Column(Float, default=0.0)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def normalize_name(name: str) -> str:
    if not name:
        return ""
    
    name = name.lower().strip()
    name = re.sub(r"[-_]", " ", name)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    # Enhanced Bengali phonetic substitutions
    substitutions = [
        (r"\b(mohamm?ad?)\b", "md"),
        (r"^[jy]", "y"),
        (r"(?<=[aeiou])[jy]", "y"),
        (r"v", "bh"),
        (r"a", "o"),
        (r"f", "ph"),
        (r"sh", "s"),
        (r"dh?", "d"),
        (r"th?", "t"),
        (r"iy", "i"),
        (r"oo", "u"),
        (r"([aeiou])h\b", r"\1"),  # Remove trailing 'h' after vowels
        (r"\bch", "c"),            # Initial 'ch' to 'c'
        (r"ck", "k")
    ]
    
    for pattern, replacement in substitutions:
        name = re.sub(pattern, replacement, name)
    
    return name

def get_phonetic_code(name: str) -> str:
    primary, secondary = doublemetaphone(name)
    return f"{primary}:{secondary}" if secondary else primary

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
    min_df=2,
    lowercase=False
)

def train_tfidf_vectorizer(db: Session, limit=100000):
    farmers = db.query(Farmer).limit(limit).all()
    details = db.query(FarmerDetails).limit(limit).all()
    
    phonetic_codes = set()
    for farmer in farmers + details:
        for field in ["name_registration", "name_aadhaar", "name_kb", "name_bank"]:
            raw_name = getattr(farmer, field)
            if raw_name:
                normalized = normalize_name(raw_name)
                if normalized:
                    code = get_phonetic_code(normalized)
                    phonetic_codes.add(code)
                    # Add alternative representations
                    phonetic_codes.add(code.replace(":", ""))

    if phonetic_codes:
        vectorizer.fit(list(phonetic_codes))
        logger.info(f"TF-IDF trained on {len(phonetic_codes)} phonetic patterns")

with SessionLocal() as db:
    train_tfidf_vectorizer(db)

def calculate_name_similarity(name1: str, name2: str, tfidf_weight: float, fuzzy_weight: float) -> float:
    norm1, norm2 = normalize_name(name1), normalize_name(name2)
    if not norm1 or not norm2:
        return 0.0
    
    try:
        # Phonetic similarity
        code1 = get_phonetic_code(norm1)
        code2 = get_phonetic_code(norm2)
        
        tfidf_matrix = vectorizer.transform([code1, code2])
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Fuzzy similarity with multiple strategies
        fuzzy_score = max(
            fuzz.token_sort_ratio(norm1, norm2),
            fuzz.partial_ratio(norm1, norm2),
            fuzz.QRatio(norm1, norm2),
            fuzz.WRatio(norm1, norm2)
        ) / 100
        
        combined_score = (tfidf_score * tfidf_weight) + (fuzzy_score * fuzzy_weight)
        return round(combined_score, 4)
    
    except Exception as e:
        logger.error(f"Similarity calculation failed: {str(e)}")
        return 0.0

app = FastAPI()

@app.post("/update_farmer_registration")
def update_farmer_registration(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.6, description="Weight for phonetic similarity"),
    fuzzy_weight: float = Query(0.4, description="Weight for string similarity")
):
    try:
        if tfidf_weight + fuzzy_weight != 1.0:
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")

        total_farmers = db.query(Farmer).filter(
            (Farmer.ai_aadhaar_name_match_flag.is_(None)) |
            (Farmer.ai_kb_name_match_flag.is_(None)) |
            (Farmer.ai_bank_name_match_flag.is_(None))
        ).count()

        logger.info(f"Processing {total_farmers} farmer registrations")
        processed = 0
        batch_size = 10000

        for offset in range(0, total_farmers, batch_size):
            farmers = db.query(Farmer).filter(
                (Farmer.ai_aadhaar_name_match_flag.is_(None)) |
                (Farmer.ai_kb_name_match_flag.is_(None)) |
                (Farmer.ai_bank_name_match_flag.is_(None))
            ).offset(offset).limit(batch_size).all()

            for farmer in farmers:
                process_farmer(farmer, db, tfidf_weight, fuzzy_weight)
                processed += 1
                
                if processed % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed {processed} records")

        if processed % 1000 != 0:
            db.commit()
            logger.info(f"Final commit: {processed} records")

        return {"message": "Farmer registration updates completed"}

    except Exception as e:
        logger.error(f"Update failed: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_farmer_details")
def update_farmer_details(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.6, description="Weight for phonetic similarity"),
    fuzzy_weight: float = Query(0.4, description="Weight for string similarity")
):
    try:
        total_details = db.query(FarmerDetails).filter(
            (FarmerDetails.ai_aadhaar_name_match_flag.is_(None)) |
            (FarmerDetails.ai_kb_name_match_flag.is_(None)) |
            (FarmerDetails.ai_bank_name_match_flag.is_(None))
        ).count()

        logger.info(f"Processing {total_details} farmer details")
        processed = 0
        batch_size = 10000

        for offset in range(0, total_details, batch_size):
            details = db.query(FarmerDetails).filter(
                (FarmerDetails.ai_aadhaar_name_match_flag.is_(None)) |
                (FarmerDetails.ai_kb_name_match_flag.is_(None)) |
                (FarmerDetails.ai_bank_name_match_flag.is_(None))
            ).offset(offset).limit(batch_size).all()

            for detail in details:
                process_farmer(detail, db, tfidf_weight, fuzzy_weight)
                processed += 1
                
                if processed % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed {processed} records")

        if processed % 1000 != 0:
            db.commit()
            logger.info(f"Final commit: {processed} records")

        return {"message": "Farmer details updates completed"}

    except Exception as e:
        logger.error(f"Update failed: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

def process_farmer(farmer, db, tfidf_weight, fuzzy_weight):
    reg_name = normalize_name(farmer.name_registration)
    if not reg_name:
        return

    for field in ["aadhaar", "kb", "bank"]:
        name_field = f"name_{field}"
        match_flag = f"ai_{field}_name_match_flag"
        match_accuracy = f"ai_{field}_name_match_accuracy"
        
        if getattr(farmer, match_accuracy) is not None:
            continue
            
        compare_name = normalize_name(getattr(farmer, name_field, ""))
        if compare_name:
            score = calculate_name_similarity(reg_name, compare_name, tfidf_weight, fuzzy_weight)
            setattr(farmer, match_accuracy, score)
            setattr(farmer, match_flag, score >= 0.45)  # Adjusted threshold

    db.add(farmer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)    

