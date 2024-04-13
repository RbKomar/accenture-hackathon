import os
from typing import List

import numpy as np
from pymilvus import (
    DataType,
    MilvusClient,
    Collection,
    AnnSearchRequest,
    RRFRanker,
    connections,
    CollectionSchema,
)
from openai import AzureOpenAI
from pymilvus.milvus_client import IndexParams

from src.create_embed_db import EmbedMilvusCreator


# Replace 'Milvus' and methods with the actual classes and methods from your Milvus SDK.
class MilvusManager:
    def __init__(self, client_uri: str):
        self.client = MilvusClient(uri=client_uri)
        self.azure_openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    def create_embedding(
        self, text_to_embed: str, model: str = "text-embedding-ada-002"
    ) -> List[float]:
        embed: List[float] = (
            self.azure_openai_client.embeddings.create(
                input=[text_to_embed], model=model
            )
            .data[0]
            .embedding
        )
        return embed

    @staticmethod
    def _prepare_schema():
        schema: CollectionSchema = MilvusClient.create_schema(
            auto_id=True, enable_dynamic_field=True
        )
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1536
        )
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )
        return schema

    @staticmethod
    def _prepare_index():
        index_params: IndexParams = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_inverted_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",  # Inner Product
            params={"drop_ratio_build": 0.2},
        )
        return index_params

    def make_collection(self, collection_name: str):
        self.client.create_collection(
            collection_name=collection_name,
            schema=self._prepare_schema(),
            index_params=self._prepare_index(),
        )

    def add_table_embedding_index(
        self,
        table_name: str,
        embedding: list[float],
        table_definition: dict,
        table_description: str,
    ):
        pass

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
        data = self.create_embedding(query)
        if output_fields is None:
            output_fields = [
                "id",
                "price",
                "cat_level_4",
                "cat_level_5",
                "url",
                "specification",
                "all_variants",
            ]

        return self.client.search(
            collection_name=collection_name,
            data=[data],
            filter=fixed_filter,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
        )

    def hybrid_search(
        self,
        collection_name: str,
        query: str = None,
        fixed_filter: str = None,
        limit: int = 2,
        output_fields: list = None,
    ):
        connections.connect(alias="default")
        collection = Collection(name=collection_name)
        res = collection.hybrid_search(
            output_fields=[
                "id",
                "price",
                "cat_level_4",
                "cat_level_5",
                "url",
                "specification",
                "all_variants",
            ],
            reqs=[
                AnnSearchRequest(
                    data=[self.create_embedding(query)],
                    anns_field="embedding",
                    param={"metric_type": "COSINE"},
                    limit=limit,
                ),
                AnnSearchRequest(
                    data=[EmbedMilvusCreator.bm25_encode(query)],
                    anns_field="sparse_vector",
                    param={
                        "metric_type": "IP",
                        "params": {
                            "drop_ratio_search": 0.2,
                        },
                    },
                    limit=limit,
                ),
            ],
            rerank=RRFRanker(),
            limit=limit,
        )

        print("Odpowiedź hybrid search: ", res)
        return res
