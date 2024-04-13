class PromptManager:

    @staticmethod
    def get_table_description_prompt(table_definition):
        prompt = """You are an SQL specialist and you are asked to describe the table below.
        After you are done with that make sure to describe what table might consist of and what might be used for.
        The table has the following columns with types: \n{columns}."""
        columns = table_definition["columns"]
        return prompt.format(columns=columns)
