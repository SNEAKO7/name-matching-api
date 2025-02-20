from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# Define the FarmerRegistration Model
class FarmerRegistration(Base):
    __tablename__ = "farmer_registration"

    id = Column(Integer, primary_key=True, index=True)
    name_aadhaar = Column(String(255))
    name_kb = Column(String(255))
    name_bank = Column(String(255))
