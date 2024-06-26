import csv

import pypyodbc as odbc


class SQLQuery:
    def __init__(self) -> None:
        self.server = 'tcp:hackaton-gr9-sqlserveri10lm.database.windows.net,1433'
        self.database = 'hackaton-gr9-sqldb'
        self.username = 'hacksqlusr012993'
        self.password = 'hacksqlusrP@ssw00rd'
        self.schema_name = 'SalesLT'
        self.connect_str = 'Driver={ODBC Driver 18 for SQL Server};Server=' + self.server + ';Database=' + self.database + ';Uid=' + self.username + ';Pwd=' + self.password + ';Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
        self.cursor = self.connect()

    def connect(self):
        conn = odbc.connect(self.connect_str)
        cursor = conn.cursor()
        return cursor

    def get_schema(self, table_name):
        sql_query = f"""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{table_name}'
        """
        self.cursor.execute(sql_query)
        schema = self.cursor.fetchall()
        return schema

    def get_table_names(self):
        sql_query = f"""
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = '{self.schema_name}'
        """
        self.cursor.execute(sql_query)
        tables = self.cursor.fetchall()
        return [table[0] for table in tables]

    def load_csv_data(self, schemas, table_names):
        with open('data/products.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write the header
            writer.writerow(["desc", "schema", "table_name", 'access'])
            # Write the rows
            for table_name, schema in zip(table_names, schemas):
                row = [f"Schema for table {table_name}", schema, table_name, "read-only"]
                writer.writerow(row)

    def main(self):
        tables = self.get_table_names()
        schemas = []
        for table in tables:
            schema = self.get_schema(table)
            schemas.append(schema)
        self.load_csv_data(schemas, tables)

    def test_query(self, sql_query):
        self.cursor.execute(sql_query)
        return self.cursor.fetchall()


if __name__ == '__main__':
    sqler = SQLQuery()
    sql_query = '''
    SELECT ProductID, Name, ProductNumber, Color, StandardCost, ListPrice, [Size], Weight, ProductCategoryID, ProductModelID, SellStartDate, SellEndDate, DiscontinuedDate
    FROM [hackaton-gr9-sqldb].SalesLT.Product
    '''
    print(sqler.test_query(sql_query))
    sqler.main()
