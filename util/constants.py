# Allowed languages
ALLOWED_LANGUAGES = ['en', 'de', 'sr']

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LEON_MODEL_ID = "LeonKogler/MergedModelAllArticles"
OLLAMA_MODEL_ID = "seyi/mergedmodels"
OLLAMA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Input:
    {input}"""