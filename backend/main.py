from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import os 
from connect import SQLQuery

app = FastAPI()

#---------------INIT DATA------------------
# storeClient = MilvusStoreWithClient(csv_file_path="../vectorstore/data/products.csv")
sqlquerer =  SQLQuery()
# COLLECTION_NAME="full_morele_pl" v

# chat = ChatOpenAI(temperature=0.5, openai_api_key=os.environ["OPENAI_API_KEY"])
#---------------TYPES------------------
class ChatMessage(BaseModel):
    message: str
    

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


#MAIN ROUTE FOR DEALING WITH CHAT MESSAGES FROM USER
@app.post("/chat", response_model=dict)
async def getMessage(message: ChatMessage):
    print("Otrzymałem wiadomość: ",message.message)
    print("Przed query: \n")
    sqlquerer.test_query('''SELECT ProductID, Name, ProductNumber, Color, StandardCost, ListPrice, [Size], Weight, ProductCategoryID, ProductModelID, SellStartDate, SellEndDate, DiscontinuedDate, ThumbNailPhoto, ThumbnailPhotoFileName, rowguid, ModifiedDate
FROM [hackaton-gr9-sqldb].SalesLT.Product''')
    assert message.message is not None, "Message must be provided"

    return {}
