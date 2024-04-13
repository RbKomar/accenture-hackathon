from pymilvus import connections, MilvusClient


class MilvusConnection:
    """Manages the connection to a Milvus server."""

    def __init__(self, host='localhost', port='19530', alias="default"):
        self.host = host
        self.port = port
        self.alias = alias

    def is_connected(self):
        """Checks if the connection to the Milvus server is active."""
        try:
            connections.get_connection(self.alias)
            return True
        except Exception:
            return False

    def connect(self):
        """Establishes a connection to the Milvus server."""
        connections.connect(alias=self.alias, host=self.host, port=self.port)
        print(f"Connected to Milvus at {self.host}:{self.port} with alias '{self.alias}'.")

    def disconnect(self):
        """Disconnects from the Milvus server."""
        connections.disconnect(alias=self.alias)
        print(f"Disconnected from Milvus with alias '{self.alias}'.")
