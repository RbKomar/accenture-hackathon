from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os
import load_dotenv

load_dotenv.load_dotenv()
#----------------------------------------------------------
# passw = os.environ.get("SQL_PASS")
class sqlAgent():
    def __init__(self,uri:str):
        self.db = SQLDatabase.from_uri(uri)
        self.schema = None
        self.query = None
        self.uri = uri
       
    #     self.llm = ChatOpenAI()
    #     self.prompt = ChatPromptTemplate("""Based on the table schema below, write a SQL query that would answer the user's question:
    #     {schema}

    #     Question: {question}
    #     SQL Query:""")
    #     self.prompt2 = ChatPromptTemplate("""Based on the table schema below, question, sql query, and sql response, write a natural language response:
    #     {schema}

    #     Question: {question}
    #     SQL Query: {query}
    #     SQL Response: {response}""")
        
    # # ----------- PART 1 (Get query to ask database)
    # def get_db_schema(self):
    #     self.db = SQLDatabase.from_uri(self.uri)
    #     self.schema = self.db.get_table_info()
    #     return self.schema
    # def schema_chain(self):
    #     chain = RunnablePassthrough.assign(schema=self.get_db_schema) | self.prompt | self.llm | StrOutputParser()
    #     return chain
    # # ----------- 

    # # ----------- PART 2 (Get all in one)
    # def run_query(self, query:str):
    #     return self.db.run(query)

    # def full_chain(self):
    #     chain = RunnablePassthrough.assign(query = self.schema_chain).assign(schema=self.get_db_schema, response = lambda vars: self.run_query(vars["query"])) | self.prompt2 | self.llm
    #     return chain
    
    def test(self):
        return self.db.run("SELECT * FROM Album LIMIT 5")

    
if __name__ == "__main__":
    user_question = 'how many albums are there in the database?'
    agent = sqlAgent(f"mysql+mysqlconnector://root:{"1234"}@localhost:3306/Chinook")

    # agent.schema_chain().invoke({"question": user_question})
    # agent.full_chain().invoke({"question": user_question})
    print(agent.test())
#show variables like port
#mysql -u root -p