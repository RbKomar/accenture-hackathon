import sys

from openai import AzureOpenAI
sys.path.append("../../src/llm/")
sys.path.append("../../src/milvus_next/")

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import dotenv

dotenv.load_dotenv(dotenv.find_dotenv())
from milvus_store import MilvusStoreWithClient
import pyodbc


class Sqlagent():
    def __init__(self, collection_name: str):
        self.llm = ChatOpenAI()
        self.milvus_store = MilvusStoreWithClient()
        self.db_name = 'hackaton-gr9-sqldb'
        self.COLLECTION_NAME = collection_name
        self.schema = None
        self.query = 'Tell me about products.'
        self.prompt = ChatPromptTemplate.from_template("""Based on the table schema name, table name, schema below,  tabels related, constraints write a SQL query that would answer the user's question:
        {schema}
        Question: {question}
        Return only sql query, nothing all.
        You always have to add [hackaton-gr9-sqldb]. before schema name, when you select from that schema, for example: SELECT * FROM [hackaton-gr9-sqldb].SalesLT.vGetAllCategories
        where in this example name of the schema is SalesLT.     

        SQL Query:""")
        self.prompt2 = ChatPromptTemplate.from_template("""Based on the table schema below, question, sql query, and sql response, write a natural language response:
        {schema}
        Question: {question}
        SQL Query: {query}
        SQL Response: {response}""")

        # Establish a connection to the SQL database
        self.conn = pyodbc.connect(
            'DRIVER={ODBC Driver 17 for SQL Server};SERVER=<server_name>;DATABASE=hackaton-gr9-sqldb;UID=<username>;PWD=<password>')

    # # ----------- PART 1 (Get query to ask database)
    def vs_search(self, *args, **kwargs):
        print("Received args:", args)
        print("Received kwargs:", kwargs)
        schema = self.milvus_store.hybrid_search(collection_name=self.COLLECTION_NAME, query=self.query)
        print("searched schema", schema)
        return schema

    def schema_chain(self):
        chain = RunnablePassthrough.assign(schema=self.vs_search) | self.prompt | self.llm | StrOutputParser()
        return chain
    # # ----------- 

    def is_enough(self):
        pass
    # # ----------- PART 2 (Get all in one)
    # def run_query(self, query:str):
    #     return self.db.run(query)

    def run_query(self, query: str):
        cursor = self.conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        cursor.close()
        return result

    def full_chain(self):
        chain = RunnablePassthrough.assign(query=self.schema_chain).assign(schema=self.vs_search,
                                                                           response=lambda vars: self.run_query(
                                                                               vars["query"])) | self.prompt2 | self.llm
        return chain


if __name__ == "__main__":
    user_question = 'Tell me about products.'
    agent = Sqlagent(collection_name="tests")
    answer = agent.schema_chain().invoke({"question": user_question})
    print("answer: ", answer)
    full_answer = agent.full_chain().invoke({"question": user_question})
    print("full_answer: ", full_answer)
