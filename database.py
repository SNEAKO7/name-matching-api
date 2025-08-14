# database.py (updated)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Hardcoded connection details
DB_USER = ""
DB_PASSWORD = ""  # Use URL encoding
DB_HOST = ""
DB_PORT = ""
DB_NAME = ""

# Construct database URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create Engine
engine = create_engine(DATABASE_URL)

# Create a Session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

