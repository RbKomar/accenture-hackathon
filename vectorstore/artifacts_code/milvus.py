import logging
from typing import Optional

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_ALIAS = "default"


class MilvusConnection:
    """Manages the connection to a Milvus server."""

    def __init__(self, host='localhost', port='19530'):
        self.host = host
        self.port = port
        self._connection = None

    def connect(self):
        """Establishes a connection to the Milvus server."""
        if self._connection is None:
            self._connection = connections.connect(alias=DEFAULT_ALIAS, host=self.host, port=self.port)
        return self._connection

    def disconnect(self):
        """Disconnects from the Milvus server."""
        if self._connection is not None:
            connections.disconnect(alias=DEFAULT_ALIAS)
            self._connection = None


class MilvusStore:
    """Manages a collection of products in a Milvus server."""

    def __init__(self, connection_manager, collection_name='store_products', dimension=128):
        self.connection = connection_manager
        self.collection_name = collection_name
        self.dimension = dimension
        self.collection: Optional[Collection] = None

    def connect(self, recreate_collection=False):
        """Connects to the Milvus server and ensures the collection exists."""
        self.connection.connect()
        if recreate_collection:
            self._drop_collection()
        self._ensure_collection_exists()

    def _drop_collection(self):
        """Drops and recreates the collection."""
        self.connection.drop_collection(self.collection_name)

    def _ensure_collection_exists(self):
        """Ensures the collection exists and is loaded."""
        if not self.connection.has_collection(self.collection_name):
            self._create_collection()
        else:
            self.collection = Collection(name=self.collection_name)
        if not self.collection.has_index():
            self._create_index()
        self.collection.load()

    def _create_collection(self):
        """Creates the collection."""
        logger.info(f"Creating collection: {self.collection_name}")
        fields = [FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
                  FieldSchema(name='store_name', dtype=DataType.VARCHAR, max_length=255),
                  FieldSchema(name='product_name', dtype=DataType.VARCHAR, max_length=255),
                  FieldSchema(name='price', dtype=DataType.FLOAT),
                  FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=self.dimension)]
        schema = CollectionSchema(fields=fields)
        self.collection = Collection(name=self.collection_name, schema=schema)
        logger.info(f"Created collection: {self.collection_name}")

    def _create_index(self):
        """Creates an index on the collection."""
        logger.info(f"Creating index: {self.collection_name}")
        index_param = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
        self.collection.create_index(field_name="embedding", index_params=index_param)
        logger.info(f"Created index: {self.collection_name}")

    def insert_data(self, store_name, product_name, price, embedding):
        """Inserts a product into the collection."""
        data = [[store_name], [product_name], [price], [embedding]]
        self.collection.insert(data)

    def filter_search(self, store_name=None, min_price=None, max_price=None):
        """Searches for products based on store name and price range."""
        expr = []
        if store_name:
            expr.append(f"store_name == '{store_name}'")
        if min_price:
            expr.append(f"price >= {min_price}")
        if max_price:
            expr.append(f"price <= {max_price}")

        if expr:
            query_expr = " and ".join(expr)
            results = self.collection.query(expr=query_expr,
                                            output_fields=["id", "store_name", "product_name", "price"])
            return results
        else:
            return []

    def embedding_search(self, embedding, filtered_ids, top_k=10):
        """Searches for similar products based on embedding."""
        if filtered_ids:
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = self.collection.search(data=[embedding], anns_field="embedding", param=search_params, limit=top_k,
                                             expr=f"id in {str(filtered_ids)}",
                                             output_fields=["id", "store_name", "product_name", "price"])
            return results
        else:
            return []


def create_milvus_store():
    """Creates a MilvusStore instance."""
    connection_manager = MilvusConnection(host='localhost', port='19530')
    milvus_store = MilvusStore(connection_manager=connection_manager, collection_name='store_products', dimension=3)
    milvus_store.connect()
    return milvus_store


def test_run():
    milvus_store = create_milvus_store()

    store_name_a = "Store A"
    store_name_b = "Store B"

    milvus_store.insert_data(store_name_b, "Product 2", 19.99, [0.4, 0.5, 0.6])
    milvus_store.insert_data(store_name_b, "Product 4", 19.9, [1.0, 1.1, 1.2])
    milvus_store.insert_data(store_name_b, "Product 5", 19.99, [1.3, 1.4, 1.5])

    milvus_store.insert_data(store_name_a, "Product 1", 9.99, [0.1, 0.2, 0.3])
    milvus_store.insert_data(store_name_a, "Product 3", 14.99, [0.7, 0.8, 0.9])
    milvus_store.insert_data(store_name_a, "Product 6", 19.99, [1.6, 1.7, 1.8])
    milvus_store.insert_data(store_name_a, "Product 7", 19.99, [1.9, 2.0, 2.1])
    milvus_store.insert_data(store_name_a, "Product 8", 19.99, [2.2, 2.3, 2.4])

    filtered_results = milvus_store.filter_search(store_name=store_name_a, min_price=5)
    filtered_ids = [result.get("id") for result in filtered_results]

    query_embedding = [1.0, 1.0, 1.2]
    embedding_results = milvus_store.embedding_search(query_embedding, filtered_ids, top_k=5)

    for results in embedding_results:
        for hit in results:
            print(f"ID: {hit.id}, Store: {hit.entity.get('store_name')}, "
                  f"Product: {hit.entity.get('product_name')}, Price: {hit.entity.get('price')}")


class MilvusStore:
    """Manages a collection in a Milvus server."""

    def __init__(self, connection_manager: MilvusConnection, csv_file_path, collection_name, embedding_dimension=128):
        self.connection_manager: MilvusConnection = connection_manager
        self.csv_loader: CSVLoader = CSVLoader(csv_file_path)
        self.collection_name = collection_name
        self.embedding_dimension = embedding_dimension
        self.embedding_generator = EmbeddingGenerator(self.embedding_dimension)
        self.collection = None
        self.logger = logging.getLogger(__name__)

    def setup_collection(self):
        """Ensures the collection and index are set up in Milvus, based on the CSV file schema."""
        self.connection_manager.connect()
        field_schemas = self.csv_loader.infer_schema_from_csv()
        if not utility.has_collection(self.collection_name):
            schema = CollectionSchema(fields=field_schemas)
            self.collection = Collection(name=self.collection_name, schema=schema)
            self.logger.info(f"Created collection: {self.collection_name}")
        else:
            self.collection = Collection(name=self.collection_name)
        if not self.collection.has_index():
            self._create_index()

    def _create_index(self):
        """Creates an index on the 'embedding' field."""
        index_param = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 1024}}
        self.collection.create_index(field_name="embedding", index_params=index_param)
        self.logger.info(f"Created index on '{self.collection_name}'.")

    def _ensure_connection(self):
        """Ensures that there is an active connection to Milvus."""
        if not self.connection_manager.is_connected():
            self.connection_manager.connect()

    def insert_data_from_csv(self):
        """Dynamically inserts data into the collection from a CSV file, based on the inferred schema."""
        self._ensure_connection()
        df = self.csv_loader.data

        for _, row in df.iterrows():
            embedding = self.embedding_generator.generate(row['long_description'])
            data = {'embedding': embedding}

            for col in df.columns:
                value = row[col]
                data[col] = value

            for field in self.collection.schema.fields:
                if field.name not in data:
                    continue
                if field.dtype == DataType.VARCHAR:
                    data[field.name] = str(data[field.name])
                elif field.dtype == DataType.FLOAT:
                    data[field.name] = float(data[field.name])
                elif field.dtype == DataType.INT64:
                    data[field.name] = int(data[field.name])
            obj = {k: v for k, v in data.items()}
            self.logger.info(f"Inserting data: {obj}")
            self.collection.insert({k: v for k, v in data.items()})

            break
        print(self.collection)


if __name__ == '__main__':
    test_run()
