import logging
import time
from random import random
from csv_loader import EmbeddingGenerator
from pymilvus import DataType, MilvusClient

logging.basicConfig(level=logging.INFO)

class Searcher:
    """Searches for data."""

    def __init__(self,client_uri:str = "http://localhost:19530", dimension=1536):
        self.dimension = dimension
        self.emebeding_model = EmbeddingGenerator.generate()
        self.client = MilvusClient(uri=client_uri)
    def search(
        self,
        collection_name: str,
        query: list = None,
        fixed_filter: str = None,
        limit: int = 100,
        output_fields: list = None,
        search_params: dict = None,
    ):
        assert query is not None, "Query must be provided"
        data = EmbeddingGenerator().generate(query)
        print ("Embedding: ", data[:10])
        if fixed_filter is None:
            # example from milvus docs: filter='claps > 30 and reading_time < 10',
            fixed_filter = "cat_level_4 == 'Akumulatory i ładowarki'"
        if output_fields is None:
            output_fields = ["id", "price","cat_level_4", "cat_level_5", "url","specification","all_variants"]
        return self.client.search(
            collection_name=collection_name,
            data=[data],
            filter=fixed_filter,
            limit=limit,
            output_fields=output_fields,
            # search_params=search_params,
        )