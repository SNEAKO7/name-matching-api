'''import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database credentials from environment variables
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "p@pyrus")  # Adjust default as needed
DB_HOST = os.getenv("DB_HOST", "callippus.co.uk")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "epaddy")

# Construct database URL dynamically
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create Engine
engine = create_engine(DATABASE_URL)

# Create a Session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()'''
# database.py (updated)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Hardcoded connection details
DB_USER = "dbadmin"
DB_PASSWORD = "dbuser%402023"  # Use URL encoding
DB_HOST = "172.25.144.165"
DB_PORT = "3306"
DB_NAME = "paddy_live"

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
