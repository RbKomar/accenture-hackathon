from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])


def main():
    embed_query = model.embed_query(text="Hello, world!")

    print(embed_query, len(embed_query))

if __name__ == "__main__":
    main()