import logging
import os
import time
from random import random

from pymilvus import DataType, MilvusClient, Collection, AnnSearchRequest, RRFRanker, connections

from csv_loader import EmbeddingGenerator
from csv_loader import CSVLoader
from setup import ROOT_FOLDER_NAME
logging.basicConfig(level=logging.INFO)


class MilvusStoreWithClient:
    def __init__(self, client_uri: str = "http://localhost:19530",
                 csv_file_path: str = r'\backend\data\products.csv'):
        project_root = os.getcwd()

        while os.path.basename(project_root) != ROOT_FOLDER_NAME:
            project_root = os.path.dirname(project_root)
            if os.path.basename(project_root) == '/':  # We've reached the file system root without finding the folder
                raise FileNotFoundError(f"The root folder '{ROOT_FOLDER_NAME}' was not found in the directory tree.")

        csv_path = project_root + csv_file_path

        self.client = MilvusClient(uri=client_uri)
        self.csv_loader = CSVLoader(csv_path)
        self.logger = logging.getLogger(__name__)

    # DO HYBRID SEARCH
    @staticmethod
    def _prepare_schema():
        schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1536)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        return schema

    @staticmethod
    def _prepare_index():
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index(field_name="sparse_vector", index_name="sparse_inverted_index",
                               index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                               params={"drop_ratio_build": 0.2})  # Inner Product
        return index_params

    def describe_collection(self, collection_name: str):
        return (self.client.describe_collection(collection_name), self.client.list_collections())

    def make_collection(self, collection_name: str):
        self.client.create_collection(collection_name=collection_name, schema=self._prepare_schema(),
            index_params=self._prepare_index(), )
        return

    # ===================

    def recreate_collection(self, collection_name: str):
        if self.client.has_collection(collection_name):
            logging.info(f"Dropping existing collection: {collection_name}")
            self.client.drop_collection(collection_name)
            while self.client.has_collection(collection_name):
                time.sleep(1)

        logging.info(f"Creating collection: {collection_name}")
        self.make_collection(collection_name)

    def insert_data_from_csv(self, collection_name: str, hybrid: bool = False):
        prepared_data = self.csv_loader.prepare_data(hybrid=hybrid)
        self.logger.info(f"Inserting {len(prepared_data)} records into collection")
        self.client.insert(collection_name=collection_name, data=prepared_data)
        time.sleep(5000)
        return

    def search(self, collection_name: str, query: list = None, fixed_filter: str = None, limit: int = 100,
            output_fields: list = None, search_params: dict = None, ):
        assert query is not None, "Query must be provided"
        data = EmbeddingGenerator().generate(query)
        if output_fields is None:
            output_fields = ["id", "price", "cat_level_4", "cat_level_5", "url", "specification", "all_variants"]

        return self.client.search(collection_name=collection_name, data=[data], filter=fixed_filter, limit=limit,
            output_fields=output_fields, # search_params=search_params,
        )

    def hybrid_search(self, collection_name: str, query: list = None, fixed_filter: str = 'access==0', limit: int = 1,
                      output_fields: list = ['schema', 'neighbours', 'schema_name', 'table_name', 'constraints']):

        connections.connect(alias="default")
        collection = Collection(name=collection_name)
        # collection.load()
        # print(len(EmbeddingGenerator().generate(query)))
        res = collection.hybrid_search(output_fields=output_fields, filter=fixed_filter,
            reqs=[AnnSearchRequest(data=[EmbeddingGenerator().generate(query)],  ## Dense vectors
                anns_field="embedding",  # Field name of the vectors
                param={"metric_type": "COSINE"}, limit=limit, ),
                AnnSearchRequest(data=[self.csv_loader.bm25_encode(query)],  ## Sparse vector
                    anns_field="sparse_vector",  # Field name of the vectors
                    param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2, }}, limit=limit, )],
            rerank=RRFRanker(), limit=limit)

        print(res[0][0].entity.get('schema_name'))

        return {"SCHEMA_NAME": res[0][0].entity.get('schema_name'), "TABLE_NAME": res[0][0].entity.get('table_name'),
                "TABLE_SCHEMA": res[0][0].entity.get("schema"),
                "IDS OF TABLES THAT ARE RELATED WITH GIVEN TABLE": res[0][0].entity.get('neighbours'),
                "TABLE CONSTRAINTS": res[0][0].entity.get('constraints')}


if __name__ == "__main__":
    # creating collection
    COLLECTION_NAME = "tests"
    milvus_store = MilvusStoreWithClient(
        csv_file_path="/Users/bartek/Documents/ai_persp/accenture-hackathon/backend/data/products.csv")

    # HYBRYDA
    # milvus_store.make_collection(COLLECTION_NAME)
    # milvus_store.insert_data_from_csv(COLLECTION_NAME, hybrid=True)
    searched_values = milvus_store.hybrid_search(COLLECTION_NAME, query="Jakie są produkty?")

    print("WYNIK: ", searched_values)
