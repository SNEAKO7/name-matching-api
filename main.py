'''import logging
import re
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = "mysql+pymysql://dbadmin:dbuser%402023@172.25.144.165:3306/paddy_live"
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20)  # Optimized DB Connection
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

# FarmerDetails model (Same structure as Farmer)
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

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Normalize names before processing
def normalize_name(name: str) -> str:
    if not name:
        return ""
    
    name = name.lower().strip()
    name = re.sub(r"[-_]", " ", name)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    
    # Replace common variations of "Mohammad"
    name = re.sub(r"\bmohammad\b", "md", name)
    name = re.sub(r"\bmohammed\b", "md", name)
    name = re.sub(r"\bmohamad\b", "md", name)
    name = re.sub(r"\bmohamed\b", "md", name)

    return name

# Train TF-IDF Vectorizer Once
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
logger.info("TF-IDF vectorizer initialized")

# Load Sample Names for TF-IDF Training
def train_tfidf_vectorizer(db: Session, limit=100000):
    farmers = db.query(Farmer).limit(limit).all()
    details = db.query(FarmerDetails).limit(limit).all()
    
    names = set()
    for farmer in farmers + details:
        if farmer.name_registration:
            names.add(normalize_name(farmer.name_registration))
        if farmer.name_aadhaar:
            names.add(normalize_name(farmer.name_aadhaar))
        if farmer.name_kb:
            names.add(normalize_name(farmer.name_kb))
        if farmer.name_bank:
            names.add(normalize_name(farmer.name_bank))
    
    if names:
        vectorizer.fit(list(names))
        logger.info(f"TF-IDF trained on {len(names)} sample names.")

# Train TF-IDF once at startup
with SessionLocal() as db:
    train_tfidf_vectorizer(db)

# Hybrid Name Similarity Calculation
def calculate_name_similarity(name1: str, name2: str, tfidf_weight: float, fuzzy_weight: float) -> float:
    name1, name2 = normalize_name(name1), normalize_name(name2)
    if not name1 or not name2:
        return 0.0
    try:
        tfidf_matrix = vectorizer.transform([name1, name2])
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        fuzzy_score = fuzz.token_sort_ratio(name1, name2) / 100
        combined_score = (tfidf_weight * tfidf_score) + (fuzzy_weight * fuzzy_score)
        return round(combined_score, 4)
    except Exception as e:
        logger.error(f"Error in similarity calculation: {str(e)}")
        return 0.0

# FastAPI App
app = FastAPI()

@app.post("/update_name_matching")
def update_name_matching(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.5, description="Weight for TF-IDF similarity"),
    fuzzy_weight: float = Query(0.5, description="Weight for Fuzzy matching")
):
    try:
        logger.info("Starting name matching update process")
        if tfidf_weight + fuzzy_weight != 1.0:
            raise HTTPException(status_code=400, detail="TF-IDF and Fuzzy weights must sum to 1.0")

        batch_size = 1000  # ✅ Reduced batch size for more frequent commits
        total_farmers = db.query(Farmer).count()
        total_details = db.query(FarmerDetails).count()

        logger.info(f"Processing {total_farmers} records from farmer_registration and {total_details} from farmer_details")

        for offset in range(0, max(total_farmers, total_details), batch_size):
            farmers = db.query(Farmer).offset(offset).limit(batch_size).all()
            details = db.query(FarmerDetails).offset(offset).limit(batch_size).all()

            for farmer in farmers:
                process_farmer_record(farmer, db, tfidf_weight, fuzzy_weight)

            for detail in details:
                process_farmer_record(detail, db, tfidf_weight, fuzzy_weight)

            db.commit()  # ✅ Commit after every 1,000 records from both tables
            logger.info(f"Committed batch up to offset {offset + batch_size}")

        logger.info("Successfully committed all name matching updates")
        return {"message": "Name matching data updated successfully"}

    except Exception as e:
        logger.error(f"Critical error during update: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

def process_farmer_record(farmer, db, tfidf_weight, fuzzy_weight):
    name_registration = normalize_name(farmer.name_registration)
    if not name_registration:
        return
    for field in ["aadhaar", "kb", "bank"]:
        name_field = f"name_{field}"
        match_flag = f"ai_{field}_name_match_flag"
        match_accuracy = f"ai_{field}_name_match_accuracy"
        name_to_compare = normalize_name(getattr(farmer, name_field, ""))
        if name_to_compare:
            accuracy = calculate_name_similarity(name_registration, name_to_compare, tfidf_weight, fuzzy_weight)
            setattr(farmer, match_accuracy, accuracy)
            setattr(farmer, match_flag, accuracy > 0.7)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)'''

#increased threshold from >0 to >0.3.
'''import logging
import re
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

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

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Normalize names
def normalize_name(name: str) -> str:
    if not name:
        return ""
    
    name = name.lower().strip()
    name = re.sub(r"[-_]", " ", name)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    # Replace common variations of "Mohammad"
    name = re.sub(r"\bmohammad\b", "md", name)
    name = re.sub(r"\bmohammed\b", "md", name)
    name = re.sub(r"\bmohamad\b", "md", name)
    name = re.sub(r"\bmohamed\b", "md", name)

    return name

# Train TF-IDF Vectorizer Once
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
logger.info("TF-IDF vectorizer initialized")

# Train TF-IDF using sample names
def train_tfidf_vectorizer(db: Session, limit=100000):
    farmers = db.query(Farmer).limit(limit).all()
    details = db.query(FarmerDetails).limit(limit).all()
    
    names = set()
    for farmer in farmers + details:
        for field in ["name_registration", "name_aadhaar", "name_kb", "name_bank"]:
            if getattr(farmer, field):
                names.add(normalize_name(getattr(farmer, field)))

    if names:
        vectorizer.fit(list(names))
        logger.info(f"TF-IDF trained on {len(names)} sample names.")

with SessionLocal() as db:
    train_tfidf_vectorizer(db)

# Hybrid Name Similarity Calculation
def calculate_name_similarity(name1: str, name2: str, tfidf_weight: float, fuzzy_weight: float) -> float:
    name1, name2 = normalize_name(name1), normalize_name(name2)
    if not name1 or not name2:
        return 0.0
    try:
        tfidf_matrix = vectorizer.transform([name1, name2])
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        fuzzy_score = fuzz.token_sort_ratio(name1, name2) / 100
        combined_score = (tfidf_weight * tfidf_score) + (fuzzy_weight * fuzzy_score)
        return round(combined_score, 4)
    except Exception as e:
        logger.error(f"Error in similarity calculation: {str(e)}")
        return 0.0

# FastAPI App
app = FastAPI()

@app.post("/update_name_matching")
def update_name_matching(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.5, description="Weight for TF-IDF similarity"),
    fuzzy_weight: float = Query(0.5, description="Weight for Fuzzy matching")
):
    try:
        logger.info("Starting name matching update process")
        if tfidf_weight + fuzzy_weight != 1.0:
            raise HTTPException(status_code=400, detail="TF-IDF and Fuzzy weights must sum to 1.0")

        batch_size = 1000
        total_farmers = db.query(Farmer).count()
        total_details = db.query(FarmerDetails).count()

        logger.info(f"Processing {total_farmers} farmers and {total_details} details")

        for offset in range(0, max(total_farmers, total_details), batch_size):
            farmers = db.query(Farmer).offset(offset).limit(batch_size).all()
            details = db.query(FarmerDetails).offset(offset).limit(batch_size).all()

            for farmer in farmers:
                process_farmer_record(farmer, db, tfidf_weight, fuzzy_weight)

            for detail in details:
                process_farmer_record(detail, db, tfidf_weight, fuzzy_weight)

            db.commit()
            logger.info(f"Committed batch up to offset {offset + batch_size}")

        logger.info("Successfully committed all name matching updates")
        return {"message": "Name matching data updated successfully"}

    except Exception as e:
        logger.error(f"Critical error during update: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Process each farmer record
def process_farmer_record(farmer, db, tfidf_weight, fuzzy_weight):
    name_registration = normalize_name(farmer.name_registration)
    if not name_registration:
        return
    for field in ["aadhaar", "kb", "bank"]:
        name_field = f"name_{field}"
        match_flag = f"ai_{field}_name_match_flag"
        match_accuracy = f"ai_{field}_name_match_accuracy"
        name_to_compare = normalize_name(getattr(farmer, name_field, ""))
        if name_to_compare:
            accuracy = calculate_name_similarity(name_registration, name_to_compare, tfidf_weight, fuzzy_weight)
            setattr(farmer, match_accuracy, accuracy)
            setattr(farmer, match_flag, accuracy > 0.3)  # ✅ Threshold increased to 0.3
            #setattr(farmer, match_flag, accuracy > 0.4)  # Threshold increased to 0.4


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)'''
'''import logging
import re
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

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

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Normalize names
def normalize_name(name: str) -> str:
    if not name:
        return ""
    
    name = name.lower().strip()
    name = re.sub(r"[-_]", " ", name)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    # Replace common variations of "Mohammad"
    name = re.sub(r"\bmohammad\b", "md", name)
    name = re.sub(r"\bmohammed\b", "md", name)
    name = re.sub(r"\bmohamad\b", "md", name)
    name = re.sub(r"\bmohamed\b", "md", name)

    return name

# Train TF-IDF Vectorizer Once
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
logger.info("TF-IDF vectorizer initialized")

# Train TF-IDF using sample names
def train_tfidf_vectorizer(db: Session, limit=100000):
    farmers = db.query(Farmer).limit(limit).all()
    details = db.query(FarmerDetails).limit(limit).all()
    
    names = set()
    for farmer in farmers + details:
        for field in ["name_registration", "name_aadhaar", "name_kb", "name_bank"]:
            if getattr(farmer, field):
                names.add(normalize_name(getattr(farmer, field)))

    if names:
        vectorizer.fit(list(names))
        logger.info(f"TF-IDF trained on {len(names)} sample names.")

with SessionLocal() as db:
    train_tfidf_vectorizer(db)

# Hybrid Name Similarity Calculation
def calculate_name_similarity(name1: str, name2: str, tfidf_weight: float, fuzzy_weight: float) -> float:
    name1, name2 = normalize_name(name1), normalize_name(name2)
    if not name1 or not name2:
        return 0.0
    try:
        tfidf_matrix = vectorizer.transform([name1, name2])
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        fuzzy_score = fuzz.token_sort_ratio(name1, name2) / 100
        combined_score = (tfidf_weight * tfidf_score) + (fuzzy_weight * fuzzy_score)
        return round(combined_score, 4)
    except Exception as e:
        logger.error(f"Error in similarity calculation: {str(e)}")
        return 0.0

# FastAPI App
app = FastAPI()

@app.post("/update_name_matching")
def update_name_matching(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.5, description="Weight for TF-IDF similarity"),
    fuzzy_weight: float = Query(0.5, description="Weight for Fuzzy matching")
):
    try:
        logger.info("Starting name matching update process")
        if tfidf_weight + fuzzy_weight != 1.0:
            raise HTTPException(status_code=400, detail="TF-IDF and Fuzzy weights must sum to 1.0")

        # We'll still fetch in larger batches per table for efficiency,
        # but commit will occur every 1,000 records processed.
        fetch_batch_size = 10000

        # Query only records that haven't been processed (i.e. at least one matching flag is null)
        total_farmers = db.query(Farmer).filter(
            (Farmer.ai_aadhaar_name_match_flag.is_(None)) |
            (Farmer.ai_kb_name_match_flag.is_(None)) |
            (Farmer.ai_bank_name_match_flag.is_(None))
        ).count()
        total_details = db.query(FarmerDetails).filter(
            (FarmerDetails.ai_aadhaar_name_match_flag.is_(None)) |
            (FarmerDetails.ai_kb_name_match_flag.is_(None)) |
            (FarmerDetails.ai_bank_name_match_flag.is_(None))
        ).count()

        logger.info(f"Processing {total_farmers} farmers and {total_details} details")

        records_processed = 0

        for offset in range(0, max(total_farmers, total_details), fetch_batch_size):
            farmers = db.query(Farmer).filter(
                (Farmer.ai_aadhaar_name_match_flag.is_(None)) |
                (Farmer.ai_kb_name_match_flag.is_(None)) |
                (Farmer.ai_bank_name_match_flag.is_(None))
            ).offset(offset).limit(fetch_batch_size).all()

            details = db.query(FarmerDetails).filter(
                (FarmerDetails.ai_aadhaar_name_match_flag.is_(None)) |
                (FarmerDetails.ai_kb_name_match_flag.is_(None)) |
                (FarmerDetails.ai_bank_name_match_flag.is_(None))
            ).offset(offset).limit(fetch_batch_size).all()

            for farmer in farmers:
                process_farmer_record(farmer, db, tfidf_weight, fuzzy_weight)
                records_processed += 1
                if records_processed % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed after processing {records_processed} records.")

            for detail in details:
                process_farmer_record(detail, db, tfidf_weight, fuzzy_weight)
                records_processed += 1
                if records_processed % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed after processing {records_processed} records.")

        # Final commit for any remaining records
        if records_processed % 1000 != 0:
            db.commit()
            logger.info(f"Final commit after processing {records_processed} records.")

        logger.info("Successfully committed all name matching updates")
        return {"message": "Name matching data updated successfully"}

    except Exception as e:
        logger.error(f"Critical error during update: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Process each farmer record, updating only the columns that are null
def process_farmer_record(farmer, db, tfidf_weight, fuzzy_weight):
    name_registration = normalize_name(farmer.name_registration)
    if not name_registration:
        return
    for field in ["aadhaar", "kb", "bank"]:
        name_field = f"name_{field}"
        match_flag = f"ai_{field}_name_match_flag"
        match_accuracy = f"ai_{field}_name_match_accuracy"
        # Update only if the matching accuracy is null (i.e. not preprocessed)
        if getattr(farmer, match_accuracy) is not None:
            continue
        name_to_compare = normalize_name(getattr(farmer, name_field, ""))
        if name_to_compare:
            accuracy = calculate_name_similarity(name_registration, name_to_compare, tfidf_weight, fuzzy_weight)
            setattr(farmer, match_accuracy, accuracy)
            setattr(farmer, match_flag, accuracy > 0.4)  # Similarity threshold updated to 0.4

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)'''

#working code latest
'''import logging
import re
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

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

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Normalize names
def normalize_name(name: str) -> str:
    if not name:
        return ""
    
    name = name.lower().strip()
    name = re.sub(r"[-_]", " ", name)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    # Replace common variations of "Mohammad"
    name = re.sub(r"\bmohammad\b", "md", name)
    name = re.sub(r"\bmohammed\b", "md", name)
    name = re.sub(r"\bmohamad\b", "md", name)
    name = re.sub(r"\bmohamed\b", "md", name)

    return name

# Train TF-IDF Vectorizer Once
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
logger.info("TF-IDF vectorizer initialized")

# Train TF-IDF using sample names
def train_tfidf_vectorizer(db: Session, limit=100000):
    farmers = db.query(Farmer).limit(limit).all()
    details = db.query(FarmerDetails).limit(limit).all()
    
    names = set()
    for farmer in farmers + details:
        for field in ["name_registration", "name_aadhaar", "name_kb", "name_bank"]:
            if getattr(farmer, field):
                names.add(normalize_name(getattr(farmer, field)))

    if names:
        vectorizer.fit(list(names))
        logger.info(f"TF-IDF trained on {len(names)} sample names.")

with SessionLocal() as db:
    train_tfidf_vectorizer(db)

# Hybrid Name Similarity Calculation
def calculate_name_similarity(name1: str, name2: str, tfidf_weight: float, fuzzy_weight: float) -> float:
    name1, name2 = normalize_name(name1), normalize_name(name2)
    if not name1 or not name2:
        return 0.0
    try:
        tfidf_matrix = vectorizer.transform([name1, name2])
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        fuzzy_score = fuzz.token_sort_ratio(name1, name2) / 100
        combined_score = (tfidf_weight * tfidf_score) + (fuzzy_weight * fuzzy_score)
        return round(combined_score, 4)
    except Exception as e:
        logger.error(f"Error in similarity calculation: {str(e)}")
        return 0.0

# FastAPI App
app = FastAPI()

@app.post("/update_farmer_registration")
def update_farmer_registration(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.5, description="Weight for TF-IDF similarity"),
    fuzzy_weight: float = Query(0.5, description="Weight for Fuzzy matching")
):
    try:
        logger.info("Starting name matching update process for farmer_registration")
        if tfidf_weight + fuzzy_weight != 1.0:
            raise HTTPException(status_code=400, detail="TF-IDF and Fuzzy weights must sum to 1.0")

        fetch_batch_size = 10000

        total_farmers = db.query(Farmer).filter(
            (Farmer.ai_aadhaar_name_match_flag.is_(None)) |
            (Farmer.ai_kb_name_match_flag.is_(None)) |
            (Farmer.ai_bank_name_match_flag.is_(None))
        ).count()

        logger.info(f"Processing {total_farmers} farmers in farmer_registration")

        records_processed = 0

        for offset in range(0, total_farmers, fetch_batch_size):
            farmers = db.query(Farmer).filter(
                (Farmer.ai_aadhaar_name_match_flag.is_(None)) |
                (Farmer.ai_kb_name_match_flag.is_(None)) |
                (Farmer.ai_bank_name_match_flag.is_(None))
            ).offset(offset).limit(fetch_batch_size).all()

            for farmer in farmers:
                process_farmer_record(farmer, db, tfidf_weight, fuzzy_weight)
                records_processed += 1
                if records_processed % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed after processing {records_processed} records.")

        if records_processed % 1000 != 0:
            db.commit()
            logger.info(f"Final commit after processing {records_processed} records.")

        logger.info("Successfully committed all name matching updates for farmer_registration")
        return {"message": "Farmer registration name matching data updated successfully"}

    except Exception as e:
        logger.error(f"Critical error during update: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_farmer_details")
def update_farmer_details(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.5, description="Weight for TF-IDF similarity"),
    fuzzy_weight: float = Query(0.5, description="Weight for Fuzzy matching")
):
    try:
        logger.info("Starting name matching update process for farmer_details")
        if tfidf_weight + fuzzy_weight != 1.0:
            raise HTTPException(status_code=400, detail="TF-IDF and Fuzzy weights must sum to 1.0")

        fetch_batch_size = 10000

        total_details = db.query(FarmerDetails).filter(
            (FarmerDetails.ai_aadhaar_name_match_flag.is_(None)) |
            (FarmerDetails.ai_kb_name_match_flag.is_(None)) |
            (FarmerDetails.ai_bank_name_match_flag.is_(None))
        ).count()

        logger.info(f"Processing {total_details} details in farmer_details")

        records_processed = 0

        for offset in range(0, total_details, fetch_batch_size):
            details = db.query(FarmerDetails).filter(
                (FarmerDetails.ai_aadhaar_name_match_flag.is_(None)) |
                (FarmerDetails.ai_kb_name_match_flag.is_(None)) |
                (FarmerDetails.ai_bank_name_match_flag.is_(None))
            ).offset(offset).limit(fetch_batch_size).all()

            for detail in details:
                process_farmer_record(detail, db, tfidf_weight, fuzzy_weight)
                records_processed += 1
                if records_processed % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed after processing {records_processed} records.")

        if records_processed % 1000 != 0:
            db.commit()
            logger.info(f"Final commit after processing {records_processed} records.")

        logger.info("Successfully committed all name matching updates for farmer_details")
        return {"message": "Farmer details name matching data updated successfully"}

    except Exception as e:
        logger.error(f"Critical error during update: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Process each farmer record, updating only the columns that are null
def process_farmer_record(farmer, db, tfidf_weight, fuzzy_weight):
    name_registration = normalize_name(farmer.name_registration)
    if not name_registration:
        return
    for field in ["aadhaar", "kb", "bank"]:
        name_field = f"name_{field}"
        match_flag = f"ai_{field}_name_match_flag"
        match_accuracy = f"ai_{field}_name_match_accuracy"
        if getattr(farmer, match_accuracy) is not None:
            continue
        name_to_compare = normalize_name(getattr(farmer, name_field, ""))
        if name_to_compare:
            accuracy = calculate_name_similarity(name_registration, name_to_compare, tfidf_weight, fuzzy_weight)
            setattr(farmer, match_accuracy, accuracy)
            setattr(farmer, match_flag, accuracy > 0.4)  # Similarity threshold updated to 0.4

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)'''



''' code with phonetics



import logging
import re
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from metaphone import doublemetaphone  # 🔹 Added for phonetic mapping

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

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Normalize names
def normalize_name(name: str) -> str:
    if not name:
        return ""
    
    name = name.lower().strip()
    name = re.sub(r"[-_]", " ", name)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    # Replace common variations of "Mohammad"
    name = re.sub(r"\bmohammad\b", "md", name)
    name = re.sub(r"\bmohammed\b", "md", name)
    name = re.sub(r"\bmohamad\b", "md", name)
    name = re.sub(r"\bmohamed\b", "md", name)

    return name

# 🔹 Phonetic Mapping (Metaphone)
def get_phonetic_representation(name: str) -> str:
    """Returns the primary phonetic representation of a name using Double Metaphone"""
    if not name:
        return ""
    return doublemetaphone(name)[0]  # Only primary key used

# Train TF-IDF Vectorizer Once
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
logger.info("TF-IDF vectorizer initialized")

# Train TF-IDF using sample names
def train_tfidf_vectorizer(db: Session, limit=100000):
    farmers = db.query(Farmer).limit(limit).all()
    details = db.query(FarmerDetails).limit(limit).all()
    
    names = set()
    for farmer in farmers + details:
        for field in ["name_registration", "name_aadhaar", "name_kb", "name_bank"]:
            if getattr(farmer, field):
                normalized_name = normalize_name(getattr(farmer, field))
                phonetic_name = get_phonetic_representation(normalized_name)
                names.add(phonetic_name)

    if names:
        vectorizer.fit(list(names))
        logger.info(f"TF-IDF trained on {len(names)} phonetic names.")

with SessionLocal() as db:
    train_tfidf_vectorizer(db)

# Hybrid Name Similarity Calculation
def calculate_name_similarity(name1: str, name2: str, tfidf_weight: float, fuzzy_weight: float) -> float:
    name1, name2 = normalize_name(name1), normalize_name(name2)
    if not name1 or not name2:
        return 0.0
    try:
        phonetic1, phonetic2 = get_phonetic_representation(name1), get_phonetic_representation(name2)
        tfidf_matrix = vectorizer.transform([phonetic1, phonetic2])
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        fuzzy_score = fuzz.token_sort_ratio(name1, name2) / 100
        combined_score = (tfidf_weight * tfidf_score) + (fuzzy_weight * fuzzy_score)
        return round(combined_score, 4)
    except Exception as e:
        logger.error(f"Error in similarity calculation: {str(e)}")
        return 0.0

# FastAPI App
app = FastAPI()

@app.post("/update_farmer_registration")
def update_farmer_registration(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.25, description="Weight for TF-IDF similarity"),
    fuzzy_weight: float = Query(0.75, description="Weight for Fuzzy matching")
):
    try:
        logger.info("Starting name matching update process for farmer_registration")
        if tfidf_weight + fuzzy_weight != 1.0:
            raise HTTPException(status_code=400, detail="TF-IDF and Fuzzy weights must sum to 1.0")

        fetch_batch_size = 10000

        total_registration = db.query(Farmer).filter(
            (Farmer.ai_aadhaar_name_match_flag.is_(None)) |
            (Farmer.ai_kb_name_match_flag.is_(None)) |
            (Farmer.ai_bank_name_match_flag.is_(None))
        ).count()

        logger.info(f"Processing {total_registration} records in farmer_registration")

        records_processed = 0

        for offset in range(0, total_registration, fetch_batch_size):
            farmers = db.query(Farmer).filter(
                (Farmer.ai_aadhaar_name_match_flag.is_(None)) |
                (Farmer.ai_kb_name_match_flag.is_(None)) |
                (Farmer.ai_bank_name_match_flag.is_(None))
            ).offset(offset).limit(fetch_batch_size).all()

            for farmer in farmers:
                process_farmer_record(farmer, db, tfidf_weight, fuzzy_weight)
                records_processed += 1
                if records_processed % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed after processing {records_processed} records.")

        if records_processed % 1000 != 0:
            db.commit()
            logger.info(f"Final commit after processing {records_processed} records.")

        logger.info("Successfully committed all name matching updates for farmer_registration")
        return {"message": "Farmer registration name matching data updated successfully"}

    except Exception as e:
        logger.error(f"Critical error during update: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_farmer_details")
def update_farmer_details(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.25, description="Weight for TF-IDF similarity"),
    fuzzy_weight: float = Query(0.75, description="Weight for Fuzzy matching")
):
    try:
        logger.info("Starting name matching update process for farmer_details")
        if tfidf_weight + fuzzy_weight != 1.0:
            raise HTTPException(status_code=400, detail="TF-IDF and Fuzzy weights must sum to 1.0")

        fetch_batch_size = 10000

        total_details = db.query(FarmerDetails).filter(
            (FarmerDetails.ai_aadhaar_name_match_flag.is_(None)) |
            (FarmerDetails.ai_kb_name_match_flag.is_(None)) |
            (FarmerDetails.ai_bank_name_match_flag.is_(None))
        ).count()

        logger.info(f"Processing {total_details} details in farmer_details")

        records_processed = 0

        for offset in range(0, total_details, fetch_batch_size):
            details = db.query(FarmerDetails).filter(
                (FarmerDetails.ai_aadhaar_name_match_flag.is_(None)) |
                (FarmerDetails.ai_kb_name_match_flag.is_(None)) |
                (FarmerDetails.ai_bank_name_match_flag.is_(None))
            ).offset(offset).limit(fetch_batch_size).all()

            for detail in details:
                process_farmer_record(detail, db, tfidf_weight, fuzzy_weight)
                records_processed += 1
                if records_processed % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed after processing {records_processed} records.")

        if records_processed % 1000 != 0:
            db.commit()
            logger.info(f"Final commit after processing {records_processed} records.")

        logger.info("Successfully committed all name matching updates for farmer_details")
        return {"message": "Farmer details name matching data updated successfully"}

    except Exception as e:
        logger.error(f"Critical error during update: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

def process_farmer_record(farmer, db, tfidf_weight, fuzzy_weight):
    try:
        for field in ["name_aadhaar", "name_kb", "name_bank"]:
            name1 = getattr(farmer, "name_registration")
            name2 = getattr(farmer, field)
            if name1 and name2:
                similarity_score = calculate_name_similarity(name1, name2, tfidf_weight, fuzzy_weight)
                ai_flag, ai_accuracy = False, 0.0
                if similarity_score >= 0.8:
                    ai_flag, ai_accuracy = True, similarity_score

                if field == "name_aadhaar":
                    farmer.ai_aadhaar_name_match_flag = ai_flag
                    farmer.ai_aadhaar_name_match_accuracy = ai_accuracy
                elif field == "name_kb":
                    farmer.ai_kb_name_match_flag = ai_flag
                    farmer.ai_kb_name_match_accuracy = ai_accuracy
                elif field == "name_bank":
                    farmer.ai_bank_name_match_flag = ai_flag
                    farmer.ai_bank_name_match_accuracy = ai_accuracy

        db.add(farmer)

    except Exception as e:
        logger.error(f"Error processing farmer record {farmer.id}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
#code for normalising down'''


''' 

this is code with notmalizing in benali


import logging
import re
from fastapi import FastAPI, Depends, HTTPException, Query
from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

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

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Normalize names
def normalize_name(name: str) -> str:
    if not name:
        return ""
    
    # Initial normalization
    name = name.lower().strip()
    name = re.sub(r"[-_]", " ", name)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    # Replace common variations of "Mohammad"
    name = re.sub(r"\bmohammad\b", "md", name)
    name = re.sub(r"\bmohammed\b", "md", name)
    name = re.sub(r"\bmohamad\b", "md", name)
    name = re.sub(r"\bmohamed\b", "md", name)

    # Bengali phonetic substitutions (fixed regex)
    # Handle J/Y at start of string
    name = re.sub(r"^[jy]", "y", name)
    # Handle J/Y after vowels
    name = re.sub(r"(?<=[aeiou])[jy]", "y", name)
    # Replace V with BH
    name = re.sub(r"v", "bh", name)
    # Replace A with O
    name = re.sub(r"a", "o", name)
    # Replace F with PH
    name = re.sub(r"f", "ph", name)

    return name

# Train TF-IDF Vectorizer Once
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
logger.info("TF-IDF vectorizer initialized")

# Train TF-IDF using sample names
def train_tfidf_vectorizer(db: Session, limit=100000):
    farmers = db.query(Farmer).limit(limit).all()
    details = db.query(FarmerDetails).limit(limit).all()
    
    names = set()
    for farmer in farmers + details:
        for field in ["name_registration", "name_aadhaar", "name_kb", "name_bank"]:
            if getattr(farmer, field):
                names.add(normalize_name(getattr(farmer, field)))

    if names:
        vectorizer.fit(list(names))
        logger.info(f"TF-IDF trained on {len(names)} sample names.")

with SessionLocal() as db:
    train_tfidf_vectorizer(db)

# Hybrid Name Similarity Calculation
def calculate_name_similarity(name1: str, name2: str, tfidf_weight: float, fuzzy_weight: float) -> float:
    name1, name2 = normalize_name(name1), normalize_name(name2)
    if not name1 or not name2:
        return 0.0
    try:
        tfidf_matrix = vectorizer.transform([name1, name2])
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        fuzzy_score = fuzz.token_sort_ratio(name1, name2) / 100
        combined_score = (tfidf_weight * tfidf_score) + (fuzzy_weight * fuzzy_score)
        return round(combined_score, 4)
    except Exception as e:
        logger.error(f"Error in similarity calculation: {str(e)}")
        return 0.0

# FastAPI App
app = FastAPI()

@app.post("/update_farmer_registration")
def update_farmer_registration(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.5, description="Weight for TF-IDF similarity"),
    fuzzy_weight: float = Query(0.5, description="Weight for Fuzzy matching")
):
    try:
        logger.info("Starting name matching update process for farmer_registration")
        if tfidf_weight + fuzzy_weight != 1.0:
            raise HTTPException(status_code=400, detail="TF-IDF and Fuzzy weights must sum to 1.0")

        fetch_batch_size = 10000

        total_farmers = db.query(Farmer).filter(
            (Farmer.ai_aadhaar_name_match_flag.is_(None)) |
            (Farmer.ai_kb_name_match_flag.is_(None)) |
            (Farmer.ai_bank_name_match_flag.is_(None))
        ).count()

        logger.info(f"Processing {total_farmers} farmers in farmer_registration")

        records_processed = 0

        for offset in range(0, total_farmers, fetch_batch_size):
            farmers = db.query(Farmer).filter(
                (Farmer.ai_aadhaar_name_match_flag.is_(None)) |
                (Farmer.ai_kb_name_match_flag.is_(None)) |
                (Farmer.ai_bank_name_match_flag.is_(None))
            ).offset(offset).limit(fetch_batch_size).all()

            for farmer in farmers:
                process_farmer_record(farmer, db, tfidf_weight, fuzzy_weight)
                records_processed += 1
                if records_processed % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed after processing {records_processed} records.")

        if records_processed % 1000 != 0:
            db.commit()
            logger.info(f"Final commit after processing {records_processed} records.")

        logger.info("Successfully committed all name matching updates for farmer_registration")
        return {"message": "Farmer registration name matching data updated successfully"}

    except Exception as e:
        logger.error(f"Critical error during update: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_farmer_details")
def update_farmer_details(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.5, description="Weight for TF-IDF similarity"),
    fuzzy_weight: float = Query(0.5, description="Weight for Fuzzy matching")
):
    try:
        logger.info("Starting name matching update process for farmer_details")
        if tfidf_weight + fuzzy_weight != 1.0:
            raise HTTPException(status_code=400, detail="TF-IDF and Fuzzy weights must sum to 1.0")

        fetch_batch_size = 10000

        total_details = db.query(FarmerDetails).filter(
            (FarmerDetails.ai_aadhaar_name_match_flag.is_(None)) |
            (FarmerDetails.ai_kb_name_match_flag.is_(None)) |
            (FarmerDetails.ai_bank_name_match_flag.is_(None))
        ).count()

        logger.info(f"Processing {total_details} details in farmer_details")

        records_processed = 0

        for offset in range(0, total_details, fetch_batch_size):
            details = db.query(FarmerDetails).filter(
                (FarmerDetails.ai_aadhaar_name_match_flag.is_(None)) |
                (FarmerDetails.ai_kb_name_match_flag.is_(None)) |
                (FarmerDetails.ai_bank_name_match_flag.is_(None))
            ).offset(offset).limit(fetch_batch_size).all()

            for detail in details:
                process_farmer_record(detail, db, tfidf_weight, fuzzy_weight)
                records_processed += 1
                if records_processed % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed after processing {records_processed} records.")

        if records_processed % 1000 != 0:
            db.commit()
            logger.info(f"Final commit after processing {records_processed} records.")

        logger.info("Successfully committed all name matching updates for farmer_details")
        return {"message": "Farmer details name matching data updated successfully"}

    except Exception as e:
        logger.error(f"Critical error during update: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

# Process each farmer record, updating only the columns that are null
def process_farmer_record(farmer, db, tfidf_weight, fuzzy_weight):
    name_registration = normalize_name(farmer.name_registration)
    if not name_registration:
        return
    for field in ["aadhaar", "kb", "bank"]:
        name_field = f"name_{field}"
        match_flag = f"ai_{field}_name_match_flag"
        match_accuracy = f"ai_{field}_name_match_accuracy"
        if getattr(farmer, match_accuracy) is not None:
            continue
        name_to_compare = normalize_name(getattr(farmer, name_field, ""))
        if name_to_compare:
            accuracy = calculate_name_similarity(name_registration, name_to_compare, tfidf_weight, fuzzy_weight)
            setattr(farmer, match_accuracy, accuracy)
            setattr(farmer, match_flag, accuracy > 0.4)  # Similarity threshold updated to 0.4

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)#code for normalising
    
    
    
    Phonetics updated code '''
'''
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

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Normalize names with Bengali phonetic rules
def normalize_name(name: str) -> str:
    if not name:
        return ""
    
    name = name.lower().strip()
    name = re.sub(r"[-_]", " ", name)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    # Replace common variations of "Mohammad"
    name = re.sub(r"\bmohammad\b", "md", name)
    name = re.sub(r"\bmohammed\b", "md", name)
    name = re.sub(r"\bmohamad\b", "md", name)
    name = re.sub(r"\bmohamed\b", "md", name)

    # Bengali phonetic substitutions
    name = re.sub(r"^[jy]", "y", name)
    name = re.sub(r"(?<=[aeiou])[jy]", "y", name)
    name = re.sub(r"v", "bh", name)
    name = re.sub(r"a", "o", name)
    name = re.sub(r"f", "ph", name)

    return name

# Phonetic representation using Double Metaphone
def get_phonetic_code(name: str) -> str:
    return doublemetaphone(name)[0] if name else ""

# Train TF-IDF Vectorizer
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
logger.info("TF-IDF vectorizer initialized")

def train_tfidf_vectorizer(db: Session, limit=100000):
    farmers = db.query(Farmer).limit(limit).all()
    details = db.query(FarmerDetails).limit(limit).all()
    
    names = set()
    for farmer in farmers + details:
        for field in ["name_registration", "name_aadhaar", "name_kb", "name_bank"]:
            raw_name = getattr(farmer, field)
            if raw_name:
                normalized = normalize_name(raw_name)
                phonetic = get_phonetic_code(normalized)
                if phonetic:
                    names.add(phonetic)

    if names:
        vectorizer.fit(list(names))
        logger.info(f"TF-IDF trained on {len(names)} phonetic names")

with SessionLocal() as db:
    train_tfidf_vectorizer(db)

# Hybrid similarity calculation
def calculate_name_similarity(name1: str, name2: str, tfidf_weight: float, fuzzy_weight: float) -> float:
    norm1, norm2 = normalize_name(name1), normalize_name(name2)
    if not norm1 or not norm2:
        return 0.0
    
    try:
        # Phonetic similarity using TF-IDF
        phonetic1 = get_phonetic_code(norm1)
        phonetic2 = get_phonetic_code(norm2)
        tfidf_matrix = vectorizer.transform([phonetic1, phonetic2])
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # String similarity using normalized names
        fuzzy_score = fuzz.token_sort_ratio(norm1, norm2) / 100
        
        return round((tfidf_score * tfidf_weight) + (fuzzy_score * fuzzy_weight), 4)
    
    except Exception as e:
        logger.error(f"Similarity calculation error: {str(e)}")
        return 0.0

# FastAPI app
app = FastAPI()

@app.post("/update_farmer_registration")
def update_farmer_registration(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.5, description="Weight for TF-IDF similarity"),
    fuzzy_weight: float = Query(0.5, description="Weight for Fuzzy matching")
):
    try:
        logger.info("Starting farmer_registration updates")
        if tfidf_weight + fuzzy_weight != 1.0:
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")

        total_farmers = db.query(Farmer).filter(
            (Farmer.ai_aadhaar_name_match_flag.is_(None)) |
            (Farmer.ai_kb_name_match_flag.is_(None)) |
            (Farmer.ai_bank_name_match_flag.is_(None))
        ).count()

        logger.info(f"Processing {total_farmers} farmers")
        records_processed = 0
        batch_size = 10000

        for offset in range(0, total_farmers, batch_size):
            farmers = db.query(Farmer).filter(
                (Farmer.ai_aadhaar_name_match_flag.is_(None)) |
                (Farmer.ai_kb_name_match_flag.is_(None)) |
                (Farmer.ai_bank_name_match_flag.is_(None))
            ).offset(offset).limit(batch_size).all()

            for farmer in farmers:
                process_record(farmer, db, tfidf_weight, fuzzy_weight)
                records_processed += 1
                
                if records_processed % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed {records_processed} records")

        if records_processed % 1000 != 0:
            db.commit()
            logger.info(f"Final commit: {records_processed} records")

        return {"message": "Farmer registration updates completed successfully"}

    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_farmer_details")
def update_farmer_details(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.5, description="Weight for TF-IDF similarity"),
    fuzzy_weight: float = Query(0.5, description="Weight for Fuzzy matching")
):
    try:
        logger.info("Starting farmer_details updates")
        if tfidf_weight + fuzzy_weight != 1.0:
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")

        total_details = db.query(FarmerDetails).filter(
            (FarmerDetails.ai_aadhaar_name_match_flag.is_(None)) |
            (FarmerDetails.ai_kb_name_match_flag.is_(None)) |
            (FarmerDetails.ai_bank_name_match_flag.is_(None))
        ).count()

        logger.info(f"Processing {total_details} details")
        records_processed = 0
        batch_size = 10000

        for offset in range(0, total_details, batch_size):
            details = db.query(FarmerDetails).filter(
                (FarmerDetails.ai_aadhaar_name_match_flag.is_(None)) |
                (FarmerDetails.ai_kb_name_match_flag.is_(None)) |
                (FarmerDetails.ai_bank_name_match_flag.is_(None))
            ).offset(offset).limit(batch_size).all()

            for detail in details:
                process_record(detail, db, tfidf_weight, fuzzy_weight)
                records_processed += 1
                
                if records_processed % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed {records_processed} records")

        if records_processed % 1000 != 0:
            db.commit()
            logger.info(f"Final commit: {records_processed} records")

        return {"message": "Farmer details updates completed successfully"}

    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

def process_record(record, db, tfidf_weight, fuzzy_weight):
    name_reg = normalize_name(record.name_registration)
    if not name_reg:
        return

    for field in ["aadhaar", "kb", "bank"]:
        name_field = f"name_{field}"
        match_flag = f"ai_{field}_name_match_flag"
        match_accuracy = f"ai_{field}_name_match_accuracy"
        
        # Skip already processed fields
        if getattr(record, match_accuracy) is not None:
            continue
            
        compare_name = normalize_name(getattr(record, name_field, ""))
        if compare_name:
            score = calculate_name_similarity(name_reg, compare_name, tfidf_weight, fuzzy_weight)
            setattr(record, match_accuracy, score)
            setattr(record, match_flag, score > 0.4)  # Original threshold

    db.add(record)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

    removed bengali normalisation''' 
'''  


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

# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Normalize names with Bengali phonetic rules
def normalize_name(name: str) -> str:
    if not name:
        return ""
    
    name = name.lower().strip()
    name = re.sub(r"[-_]", " ", name)
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    # Replace common variations of "Mohammad"
    name = re.sub(r"\bmohammad\b", "md", name)
    name = re.sub(r"\bmohammed\b", "md", name)
    name = re.sub(r"\bmohamad\b", "md", name)
    name = re.sub(r"\bmohamed\b", "md", name)

    # Bengali phonetic substitutions
    #name = re.sub(r"^[jy]", "y", name)
   # name = re.sub(r"(?<=[aeiou])[jy]", "y", name)
    #name = re.sub(r"v", "bh", name)
   # name = re.sub(r"a", "o", name)
   # name = re.sub(r"f", "ph", name)

    return name

# Phonetic representation using Double Metaphone
def get_phonetic_code(name: str) -> str:
    return doublemetaphone(name)[0] if name else ""

# Train TF-IDF Vectorizer
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
logger.info("TF-IDF vectorizer initialized")

def train_tfidf_vectorizer(db: Session, limit=100000):
    farmers = db.query(Farmer).limit(limit).all()
    details = db.query(FarmerDetails).limit(limit).all()
    
    names = set()
    for farmer in farmers + details:
        for field in ["name_registration", "name_aadhaar", "name_kb", "name_bank"]:
            raw_name = getattr(farmer, field)
            if raw_name:
                normalized = normalize_name(raw_name)
                phonetic = get_phonetic_code(normalized)
                if phonetic:
                    names.add(phonetic)

    if names:
        vectorizer.fit(list(names))
        logger.info(f"TF-IDF trained on {len(names)} phonetic names")

with SessionLocal() as db:
    train_tfidf_vectorizer(db)

# Hybrid similarity calculation
def calculate_name_similarity(name1: str, name2: str, tfidf_weight: float, fuzzy_weight: float) -> float:
    norm1, norm2 = normalize_name(name1), normalize_name(name2)
    if not norm1 or not norm2:
        return 0.0
    
    try:
        # Phonetic similarity using TF-IDF
        phonetic1 = get_phonetic_code(norm1)
        phonetic2 = get_phonetic_code(norm2)
        tfidf_matrix = vectorizer.transform([phonetic1, phonetic2])
        tfidf_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # String similarity using normalized names
        fuzzy_score = fuzz.token_sort_ratio(norm1, norm2) / 100
        
        return round((tfidf_score * tfidf_weight) + (fuzzy_score * fuzzy_weight), 4)
    
    except Exception as e:
        logger.error(f"Similarity calculation error: {str(e)}")
        return 0.0

# FastAPI app
app = FastAPI()

@app.post("/update_farmer_registration")
def update_farmer_registration(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.5, description="Weight for TF-IDF similarity"),
    fuzzy_weight: float = Query(0.5, description="Weight for Fuzzy matching")
):
    try:
        logger.info("Starting farmer_registration updates")
        if tfidf_weight + fuzzy_weight != 1.0:
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")

        total_farmers = db.query(Farmer).filter(
            (Farmer.ai_aadhaar_name_match_flag.is_(None)) |
            (Farmer.ai_kb_name_match_flag.is_(None)) |
            (Farmer.ai_bank_name_match_flag.is_(None))
        ).count()

        logger.info(f"Processing {total_farmers} farmers")
        records_processed = 0
        batch_size = 10000

        for offset in range(0, total_farmers, batch_size):
            farmers = db.query(Farmer).filter(
                (Farmer.ai_aadhaar_name_match_flag.is_(None)) |
                (Farmer.ai_kb_name_match_flag.is_(None)) |
                (Farmer.ai_bank_name_match_flag.is_(None))
            ).offset(offset).limit(batch_size).all()

            for farmer in farmers:
                process_record(farmer, db, tfidf_weight, fuzzy_weight)
                records_processed += 1
                
                if records_processed % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed {records_processed} records")

        if records_processed % 1000 != 0:
            db.commit()
            logger.info(f"Final commit: {records_processed} records")

        return {"message": "Farmer registration updates completed successfully"}

    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_farmer_details")
def update_farmer_details(
    db: Session = Depends(get_db),
    tfidf_weight: float = Query(0.5, description="Weight for TF-IDF similarity"),
    fuzzy_weight: float = Query(0.5, description="Weight for Fuzzy matching")
):
    try:
        logger.info("Starting farmer_details updates")
        if tfidf_weight + fuzzy_weight != 1.0:
            raise HTTPException(status_code=400, detail="Weights must sum to 1.0")

        total_details = db.query(FarmerDetails).filter(
            (FarmerDetails.ai_aadhaar_name_match_flag.is_(None)) |
            (FarmerDetails.ai_kb_name_match_flag.is_(None)) |
            (FarmerDetails.ai_bank_name_match_flag.is_(None))
        ).count()

        logger.info(f"Processing {total_details} details")
        records_processed = 0
        batch_size = 10000

        for offset in range(0, total_details, batch_size):
            details = db.query(FarmerDetails).filter(
                (FarmerDetails.ai_aadhaar_name_match_flag.is_(None)) |
                (FarmerDetails.ai_kb_name_match_flag.is_(None)) |
                (FarmerDetails.ai_bank_name_match_flag.is_(None))
            ).offset(offset).limit(batch_size).all()

            for detail in details:
                process_record(detail, db, tfidf_weight, fuzzy_weight)
                records_processed += 1
                
                if records_processed % 1000 == 0:
                    db.commit()
                    logger.info(f"Committed {records_processed} records")

        if records_processed % 1000 != 0:
            db.commit()
            logger.info(f"Final commit: {records_processed} records")

        return {"message": "Farmer details updates completed successfully"}

    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

def process_record(record, db, tfidf_weight, fuzzy_weight):
    name_reg = normalize_name(record.name_registration)
    if not name_reg:
        return

    for field in ["aadhaar", "kb", "bank"]:
        name_field = f"name_{field}"
        match_flag = f"ai_{field}_name_match_flag"
        match_accuracy = f"ai_{field}_name_match_accuracy"
        
        # Skip already processed fields
        if getattr(record, match_accuracy) is not None:
            continue
            
        compare_name = normalize_name(getattr(record, name_field, ""))
        if compare_name:
            score = calculate_name_similarity(name_reg, compare_name, tfidf_weight, fuzzy_weight)
            setattr(record, match_accuracy, score)
            setattr(record, match_flag, score >= 0.3)  # Original threshold

    db.add(record)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    updated code for more accuracy with bengali advanced normalization
    '''
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
