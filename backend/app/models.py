from sqlalchemy import Column, Integer, Float, String, DateTime
from datetime import datetime
from backend.app import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class TransactionRecord(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_amount = Column(Float)
    transaction_time = Column(Integer)
    customer_age = Column(Integer)
    merchant_id = Column(Integer)
    customer_location = Column(Integer)
    fraud_probability = Column(Float)
    prediction = Column(Integer)
    risk_level = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)