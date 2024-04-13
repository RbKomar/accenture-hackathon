import os
from typing import List, Dict, Any, Tuple, Self

import numpy as np
from sqlalchemy import create_engine, MetaData, Table

from src.llm.llm_manager import LLMManager
from src.milvus.milvus_manager import MilvusManager
from pinecone_text.sparse import BM25Encoder


class EmbedMilvusCreator:
    TABLES_WITH_EXPLANATIONS = dict()
    bm25 = None

    def __init__(self, db_uri: str):
        self.db_engine = create_engine(db_uri)
        self.meta_data = MetaData()
        self.milvus_manager = MilvusManager(os.environ.get("MILVUS_URL_STRING"))
        self.llm_manager = LLMManager()

    def train_bm25_encoder(
        self,
    ):
        # TODO: train bm25 over table
        self.bm25 = BM25Encoder()

    def get_db_schema(self) -> List[Dict[str, dict]]:
        self.meta_data.reflect(bind=self.db_engine)
        schema = []
        for table_name, table in self.meta_data.tables.items():
            table_info = {
                "name": table_name,
                "columns": {column.name: str(column.type) for column in table.columns},
            }
            schema.append(table_info)
        return schema

    def create_llm_explanation_of_tables(self, tables: List[Dict[str, Any]]):
        for table in tables:
            table_name = table["name"]
            table_definition = table["columns"]
            table_description: str = self.llm_manager.generate_table_description(
                table_definition
            )
            self.TABLES_WITH_EXPLANATIONS[table_name] = (
                table_definition,
                table_description,
            )
        return self

    def add_embeddings(self) -> Dict[str, Tuple[List[float], Any, str]]:
        tables_with_embeds = {}
        for table_name, (
            table_definition,
            table_description,
        ) in self.TABLES_WITH_EXPLANATIONS.items():
            table_embedding = self.milvus_manager.create_embedding(table_description)
            tables_with_embeds[table_name] = (
                table_embedding,
                table_definition,
                table_description,
            )
        return tables_with_embeds

    def build_tables_with_embeds(self) -> Dict[Any, Tuple[List[float], Any, str]]:
        tables = self.get_db_schema()
        tables_with_embeds = self.create_llm_explanation_of_tables(
            tables
        ).add_embeddings()
        # TODO: save to csv
        return tables_with_embeds

def transform_to_dict(self):
    output = {

    }