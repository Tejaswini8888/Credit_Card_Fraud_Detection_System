from pydantic import BaseModel

class UserCreate(BaseModel):
    username: str
    password: str

class Transaction(BaseModel):
    transaction_amount: float
    transaction_time: int
    customer_age: int
    merchant_id: int
    customer_location: int