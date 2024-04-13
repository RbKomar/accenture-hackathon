from random import random
import pandas as pd
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
from pymilvus import DataType, MilvusClient
from pinecone_text.sparse import BM25Encoder
from raptor import Raptor
import statistics

class EmbeddingGenerator:
    """Generates embeddings for textual data."""

    def __init__(self, dimension=1536):
        self.dimension = dimension
        self.emebeding_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    

    def generate(self, text):
        """Generates an embedding for the text."""
        return self.emebeding_model.embed_query(text=text)


class CSVLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.client = "" #MilvusClient(uri="http://localhost:19530")
        self.bm25 = BM25Encoder()
        self.raptor = Raptor()
        self.data = self.load_data()

    def load_data(self):
        data = pd.read_csv(self.file_path)
        # data = data[data["cat_level_2"] == "Laptopy"]
        print (data.shape, "shape")
        data = data.head(20)
        data_descriptions = data["long_description"].values.tolist()
        # data_categories = data["cat_level_2"].values.tolist()

        #when there is a missing value in the description
        self.fillna_type_compliant(data)

        #CHECK DATA LENGTH
        filtered_descriptions = [x for x in data_descriptions if isinstance(x, str)]

        #RAPTOR THINGS
        # docs_length = [self.raptor.num_of_tokens_for_doc(desc,"gpt-4") for desc in filtered_descriptions]
        # print("Descriptions length: ",docs_length, "\nMediana: ",statistics.median(docs_length), "\nWartość średnia: ",statistics.mean(docs_length), "\nMax: ",max(docs_length), "\nMin: ",min(docs_length))
        # raptor_desriptions_tree = self.raptor.recursive_embed_cluster_summarize(filtered_descriptions)
        # print("Wynik: ", raptor_desriptions_tree)

        #FIT BM25
        self.bm25.fit(data['long_description'].values.tolist())

        return data

    @staticmethod
    def fillna_type_compliant(df):
        for column in df.columns:
            dtype = df[column].dtype
            if pd.api.types.is_integer_dtype(dtype):
                df[column] = df[column].fillna(0)
            elif pd.api.types.is_float_dtype(dtype):
                df[column] = df[column].fillna(0.0)
            elif pd.api.types.is_string_dtype(dtype):
                df[column] = df[column].fillna("missing-data")
            elif pd.api.types.is_bool_dtype(dtype):
                df[column] = df[column].fillna(False)

    # POD DODANIE 100k rekordów z zabezpieczeniem na szybko
    @staticmethod
    def _prepare_schema():
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1536)
        return schema
    @staticmethod
    def _prepare_index():
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="L2")
        return index_params
    #------------------
    def bm25_encode(self, text):
        list_sparse = self.bm25.encode_documents(text)
        dict_sparse = dict(zip(list_sparse['indices'], list_sparse['values']))
        # print("dict sparse: ",dict_sparse)
        return dict_sparse
    
    def prepare_data(
        self,
        embedding_generator: EmbeddingGenerator = EmbeddingGenerator(),
        embedding_field_name="long_description",
        exclude_fields=None,
        hybrid  = False
    ):
        if exclude_fields is None:
            exclude_fields = ["id"]

        print ("data in prepare ds: ", self.data.keys())

        data_rows = []
        for _, row in self.data.iterrows():
            prepared_row = {}
            # print("WIERSZ: " ,row, "\n","-----------","\n")
            for column in self.data.columns:
                if column not in exclude_fields:
                    if column == embedding_field_name:
                        prepared_row["embedding"] = embedding_generator.generate(row[column])
                        if(hybrid): prepared_row["sparse_vector"] = self.bm25_encode(row[column])
                    else:
                        prepared_row[column] = row[column]
            data_rows.append(prepared_row)

        # print("START: ",data_rows[:3])
        # print("END: ",data_rows[0]["sparse_vector"])
        return data_rows



    def process_row(self,row, embedding_generator, embedding_field_name, exclude_fields):
        prepared_row = {}
        for column in row.index:
            if column not in exclude_fields:
                if column == embedding_field_name:
                    prepared_row["embedding"] = embedding_generator.generate(row[column])
                    prepared_row[column] = row[column]
                else:
                    prepared_row[column] = row[column]
        return prepared_row

    def insert_data(self,client, collection_name, data,i):
        print(f"Inserting data: ",i)
        client.insert(collection_name=collection_name, data=data)

    def vector_store_creation(
        self,
        embedding_generator: EmbeddingGenerator = EmbeddingGenerator(),
        embedding_field_name="long_description",
        exclude_fields=None,
        collection_name=None,
        num_threads=7
    ):
        assert collection_name is not None, "Collection name must be provided"
        if exclude_fields is None:
            exclude_fields = ["id"]
        data_rows = []

        def visualize_thread_operation(operation, item):
            thread_id = threading.get_ident()  # Gets the current thread's identifier
            print(f"{threading.current_thread().name} (ID: {thread_id}): {operation} {item}")
        # POD DODANIE 100k rekordów z zabezpieczeniem na szybko
        self.client.create_collection(
            collection_name=collection_name,
            schema=self._prepare_schema(),
            index_params=self._prepare_index(),
        )
        #------------------
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Dispatch phase
            result=[]
            futures = [executor.submit(self.process_row, row, embedding_generator, embedding_field_name, exclude_fields) 
                       for chunk in [self.data[i:i+1000] for i in range(0, len(self.data), 1000)] 
                       for _, row in chunk.iterrows()]
            
            # Collection phase
            start = time.time()
            for i, future in enumerate(futures):
                result.append(future.result())  
                if i % 1000 == 0:  
                    visualize_thread_operation("processing next chunk ", i // 1000 + 1)
                    self.insert_data(self.client, collection_name, result, i // 1000 + 1)
                    result=[]
                    end = time.time()
                    print("Czas na chunk: ", end-start)
            if result:
                visualize_thread_operation("processing final chunk ", "...")
                self.insert_data(self.client, collection_name, result, "final")

        return 1    

    

if __name__ == "__main__":
    print("Running the CSVLoader")
    csv_loader = CSVLoader("data/products_all.csv")
    prepared_data = csv_loader.prepare_data()
    # csv_loader.load_data()
    # csv_loader.vector_store_creation(collection_name="full_morele_pl")
    # Fix the printing function to nicely format the output
    import json
    # print(json.dumps(prepared_data, indent=2))
