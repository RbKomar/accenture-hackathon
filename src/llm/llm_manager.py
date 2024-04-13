import os

import dotenv
from openai.types import Completion
from openai.types.chat import ChatCompletion

from src.llm.prompt_manager import PromptManager
from openai import AzureOpenAI, Stream


class LLMManager:
    def __init__(self):
        self.azure_openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.prompt_manager = PromptManager()
        dotenv.load_dotenv()

    def generate_table_description(self, table_definition) -> str:
        prompt = self.prompt_manager.get_table_description_prompt(table_definition)
        response: ChatCompletion | Stream[ChatCompletion] = (
            self.azure_openai_client.chat.completions.create(
                model="gpt-35-t",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.7,
                max_tokens=800,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
        )

        description: str = response.choices[0].message.content.strip()
        return description
