# This file will be executed when a user wants to query your project.
import argparse
import json
import logging
import os
from os.path import join
from typing import List

from langchain.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.documents import Document
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate

from util import constants
from util.library import util_common


def generate_query_tags(query_text):
    empty_prompt = constants.OLLAMA_PROMPT

    instruction = "You are a helpful assistant working at the EU. It is your job to give users unbiased article recommendations. To do so, you always provide a list of tags, whenever you are prompted with a search query. The tags should represent the core ideas of user search query, and always be unbiased and in English. Respond with the tags only, separated by commas in a string array. Each tag can be 1 or more words, if needed"

    prompt = PromptTemplate.from_template(empty_prompt)
    llm = Ollama(model=constants.OLLAMA_MODEL_ID)
    chain = prompt | llm

    response = chain.invoke({"instruction": instruction, "input": query_text})
    print(response)

    tags = response.split(",")
    tags = list(set(map(lambda x: x.strip().lower(), tags)))

    return tags


# TODO Implement the inference logic here
def handle_user_query(query, query_id, output_path):
    query_lang = util_common.detect_language(query)
    if not util_common.is_allowed_language(query_lang):
        logging.warning(f"Query language '{query_lang}' is not explicitly supported. Results may be suboptimal. Supported languages are: {constants.ALLOWED_LANGUAGES}")

    translated_query = query if query_lang == "en" else util_common.translate_query(query, query_lang)

    docs = []

    for file_name in os.listdir(output_path):
        if file_name.endswith('.json'):  # Ensure we are only processing JSON files
            file_path = os.path.join(output_path, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
                tags = data['transformed_representation']
                doc = Document(
                    page_content=', '.join(tags),
                    metadata={"file_name": file_name, "source": file_path}
                )
                docs.append(doc)

    embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

    # load it into Chroma
    vector_db = Chroma.from_documents(docs, embedding_function)

    generated_queries = []

    # Output parser will split the LLM result into a list of queries
    class LineListOutputParser(BaseOutputParser[List[str]]):
        """Output parser for a list of lines."""
        def parse(self, text: str) -> List[str]:
            lines = text.strip().split("\n")
            lines_tags = []
            for line in lines:
                generated_queries.append(line)
                line_tag = generate_query_tags(line)
                lines_tags.append(', '.join(line_tag))

            return lines_tags

    output_parser = LineListOutputParser()

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five 
               different versions of the given user question to retrieve relevant documents from a vector 
               database. By generating multiple perspectives on the user question, your goal is to help
               the user overcome some of the limitations of the distance-based similarity search. 
               Provide these alternative questions separated by newlines. Response with the alternative questions only.
               Original question: {question}""",
    )
    llm = Ollama(model=constants.OLLAMA_MODEL_ID)

    # Chain
    llm_chain = QUERY_PROMPT | llm | output_parser

    # import logging
    #
    # logging.basicConfig()
    # logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

    # Run
    retriever = MultiQueryRetriever(
        retriever=vector_db.as_retriever(), llm_chain=llm_chain, include_original=True)

    # Results
    unique_docs = retriever.invoke(translated_query)
    len(unique_docs)

    result = {
        "generated_queries": generated_queries,
        "detected_language": query_lang,
    }

    with open(join(output_path, f"{query_id}.json"), "w") as f:
        json.dump(result, f)

    search_results = {
        "query": query,
        "results": [
            {
                "index": index,
                "file": doc.metadata["file_name"],
                "source": doc.metadata["source"],
                "article_tags": doc.page_content
            }
            for index, doc in enumerate(unique_docs)
        ]
    }

    print(search_results)


if True:
    handle_user_query("What are the benefits of LLMs in programming?", "1", "output")
    #rank_articles(["What are the benefits of LLMs in programming?"], [[ "llms", "ai", "programming" ], [ "war in ukraine", "russia", "ukraine"]])
    exit(0)

exit(0)

# This is a sample argparse-setup, you probably want to use in your project:
parser = argparse.ArgumentParser(description='Run the inference.')
parser.add_argument('--query', type=str, help='The user query.', required=True, action="append")
parser.add_argument('--query_id', type=str, help='The IDs for the queries, in the same order as the queries.',
                    required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    queries = args.query
    query_ids = args.query_id
    output = args.output

    assert len(queries) == len(query_ids), "The number of queries and query IDs must be the same."

    for query, query_id in zip(queries, query_ids):
        handle_user_query(query, query_id, output)
