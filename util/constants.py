# Allowed languages
ALLOWED_LANGUAGES = ['en', 'de', 'sr']

PRE_PROCESSED_PATH = "data/preprocessed"
LEON_MODEL_ID = "LeonKogler/MergedModelAllArticles"
OLLAMA_MODEL_ID = "seyi/mergedmodels"
OLLAMA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input}"""

MULTI_QUERY_PROMPT = """You are an AI language model assistant. Your task is to generate five 
               different versions of the given user question to retrieve relevant documents from a vector 
               database. By generating multiple perspectives on the user question, your goal is to help
               the user overcome some of the limitations of the distance-based similarity search. 
               Provide these alternative questions separated by newlines. Respond with the alternative questions only.
               The first line should be be the header: Alternative Questions
               Original question: {question}"""