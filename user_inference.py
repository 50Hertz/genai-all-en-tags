# This file will be executed when a user wants to query your project.
import argparse
from os.path import join
import json
import os
import requests as r
from langchain.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from typing import List
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import CharacterTextSplitter
from pydantic import BaseModel, Field
from util.library import util_common
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate


def generate_query_tags(query):
    empty_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Input:
        {input}"""

    hf = HuggingFacePipeline.from_model_id(
        model_id="LeonKogler/MergedModelAllArticles",
        task="text-generation",
        device_map="auto",
        pipeline_kwargs={"max_new_tokens": 32},
    )

    instruction = "You are a helpful assistant working at the EU. It is your job to give users unbiased article recommendations. To do so, you always provide a list of tags, whenever you are prompted with a search query. The tags should represent the core ideas of user search query, and always be unbiased and in English. Respond with the tags only, separated by commas in a string array."

    prompt = PromptTemplate.from_template(empty_prompt)
    chain = prompt | hf

    response = chain.invoke({"instruction": instruction, "input": query})
    print(response)

    tags = response.split(",")
    tags = list(set(map(lambda x: x.strip().lower(), tags)))

    return tags


# TODO Implement the inference logic here
def handle_user_query(query, query_id, output_path):

    query_lang = util_common.detect_language(query)
    if not util_common.is_allowed_language(query_lang):
        raise Exception(f"Language {query_lang} is not allowed.")

    translated_query = util_common.translate_query(query, query_lang)

    docs = []

    for file_name in os.listdir("data/transformed"):
        if file_name.endswith('.json'):  # Ensure we are only processing JSON files
            file_path = os.path.join("data/transformed", file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
                tags = data["transformed_representation"]
                doc = Document(
                    page_content=', '.join(tags),
                    metadata={"file_name": file_name, "source": file_path}
                )

                docs.append(doc)

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into Chroma
    vector_db = Chroma.from_documents(docs, embedding_function)

    # Output parser will split the LLM result into a list of queries
    class LineList(BaseModel):
        # "lines" is the key (attribute name) of the parsed output
        lines: List[str] = Field(description="Lines of text")

    class LineListOutputParser(PydanticOutputParser):
        def __init__(self) -> None:
            super().__init__(pydantic_object=LineList)

        def parse(self, text: str) -> LineList:
            lines = text.strip().split("\n")
            lines_tags = []
            for line in lines:
                line_tag = generate_query_tags(line)
                lines_tags.append(', '.join(line_tag))

            return LineList(lines=lines_tags)

    output_parser = LineListOutputParser()

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
               different versions of the given user question to retrieve relevant documents from a vector 
               database. By generating multiple perspectives on the user question, your goal is to help
               the user overcome some of the limitations of the distance-based similarity search. 
               Provide these alternative questions separated by newlines.
               Original question: {question}""",
    )
    llm = ChatOpenAI(temperature=0)

    # Chain
    llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)

    # Run
    retriever = MultiQueryRetriever(
        retriever=vector_db.as_retriever(), llm_chain=llm_chain, parser_key="lines"
    )  # "lines" is the key (attribute name) of the parsed output

    # Results
    unique_docs = retriever.invoke(query=translated_query)
    len(unique_docs)




    call_data = {
        "model": "llama3",
        "prompt": "User Query: " + query,
        "system": "You are a helpful assistant working at the EU. It is your job to give users unbiased article recommendations. You will be prompted with a user generated search query. Try to improve it by adding more context, such that a recommendation system has an easier time to find relevant articles. Response with the improved query only. The query should be unbiased and in English. The query is given as the next message. Only answer with the improved query. After the query, append LANGUAGE: [lang_code] where you use the default two-letter language code in a new line.",
        "stream": False
    }
    response = r.post("http://localhost:11434/api/generate", json=call_data)
    response = response.json()
    response_text = response["response"]
    
    new_query, language_code = response_text.split("LANGUAGE: ")
    new_query = new_query.strip()
    language_code = language_code.strip()
    
    result = {
        "generated_queries": [ new_query ],
        "detected_language": language_code,
    }
    
    print(response_text)
    
    with open(join(output_path, f"{query_id}_result.json"), "w") as f:
        json.dump(result, f)


# if True:
#     # handle_user_query("What are the benefits of LLMs in programming?", "1", "output")
#     rank_articles(["What are the benefits of LLMs in programming?"], [[ "llms", "ai", "programming" ], [ "war in ukraine", "russia", "ukraine"]])
#     exit(0)
#
# exit(0)



# This is a sample argparse-setup, you probably want to use in your project:
parser = argparse.ArgumentParser(description='Run the inference.')
parser.add_argument('--query', type=str, help='The user query.', required=True, action="append")
parser.add_argument('--query_id', type=str, help='The IDs for the queries, in the same order as the queries.', required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    queries = args.query
    query_ids = args.query_id
    output = args.output
    
    assert len(queries) == len(query_ids), "The number of queries and query IDs must be the same."
    
    for query, query_id in zip(queries, query_ids):
        handle_user_query(query, query_id, output)
    