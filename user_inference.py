# This file will be executed when a user wants to query your project.
import asyncio
import json
import logging
import os
import shutil
from os.path import join
from pprint import pprint
from typing import List

from langchain.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from util import constants
from util.library import util_common


def generate_query_tags(query_text):
    empty_prompt = constants.OLLAMA_PROMPT

    instruction = """You are a helpful assistant working at the EU. It is your job to give users unbiased article 
    recommendations. To do so, you always provide a list of tags, whenever you are prompted with a user search query. 
    These tags should represent the core ideas of the user search query, and always be unbiased and in English.
    However, some of the generated tags may be non-English only if they are proper nouns like names or places. Each  
    tag may be 1 word or a phrase of maximum 8 words. Respond with only the tags, separated by new line characters Do 
    not describe the query in any detail in the response, regardless of the language. Always respond with just the 
    tags. The first line should be be the header: Tags"""

    prompt = PromptTemplate.from_template(empty_prompt)
    llm = Ollama(model=constants.OLLAMA_MODEL_ID)
    chain = prompt | llm

    response = chain.invoke({"instruction": instruction, "input": query_text})
    #print(response)

    tags = response.split("\n")
    tags = list(filter(None, tags))
    tags = tags[1:]
    tags = list(set(map(lambda x: x.strip(), tags)))

    return tags


# TODO Implement the inference logic here
def handle_user_query(query, query_id, output_path):
    query_lang = util_common.detect_language(query)
    if not util_common.is_allowed_language(query_lang):
        logging.warning(
            f"Query language '{query_lang}' is not explicitly supported. Results may be suboptimal. Supported languages are: {constants.ALLOWED_LANGUAGES}")

    translated_query = query if query_lang == "en" else util_common.translate_query(query, query_lang)

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=constants.MULTI_QUERY_PROMPT,
    )
    llm = Ollama(model=constants.OLLAMA_MODEL_ID)

    # Chain
    chain = QUERY_PROMPT | llm

    response = chain.invoke({"question": translated_query})

    generated_queries = response.strip().split("\n")
    #Remove empty strings
    generated_queries = list(filter(None, generated_queries))
    generated_queries = list(map(lambda x: x.strip(), generated_queries))

    list_size = len(generated_queries)
    if list_size > 5:
        generated_queries = generated_queries[-5:]

    os.makedirs(output_path, exist_ok=True)
    with open(join(output_path, f"{query_id}.json"), "w") as f:
        json.dump({
            "generated_queries": generated_queries,
            "detected_language": query_lang,
        }, f)

    asyncio.run(perform_search(translated_query, generated_queries))


# Subclass of MultiQueryRetriever
class CustomMultiQueryRetriever(MultiQueryRetriever):
    generated_queries: List[str] = []

    def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        queries = self.generated_queries
        if self.include_original:
            queries.append(query)
        documents = self.retrieve_documents(queries, run_manager)
        return self.unique_union(documents)


async def perform_search(translated_query, generated_queries):
    original_query_tags = generate_query_tags(translated_query)
    query_tag_list = []
    for line in generated_queries:
        line_tag = generate_query_tags(line)
        query_tag_list.append(', '.join(line_tag))

    docs = []

    os.makedirs(constants.PRE_PROCESSED_PATH, exist_ok=True)
    for file_name in os.listdir(constants.PRE_PROCESSED_PATH):
        if file_name.endswith('.json'):  # Ensure we are only processing JSON files
            file_path = os.path.join(constants.PRE_PROCESSED_PATH, file_name)
            with open(file_path, 'r') as file:
                data = json.load(file)
                tags = data.get('transformed_representation', [])
                if not tags:
                    continue

                doc = Document(
                    page_content=', '.join(tags),
                    metadata={'file_name': file_name, 'source': file_path, 'article_file': f"data/articles/{file_name}"}
                )
                docs.append(doc)

                embedded_directory = constants.PRE_PROCESSED_PATH + '/embedded'
                os.makedirs(embedded_directory, exist_ok=True)
                # Move the file to the destination folder after processing
                shutil.move(file_path, os.path.join(embedded_directory, file_name))

    embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

    vector_db = Chroma(
        collection_name="articles",
        embedding_function=embedding_function,
        persist_directory="./chroma",
        collection_metadata={"hnsw:space": "cosine"}
    )

    if docs:
        vector_db.add_documents(docs)
        print("Documents loaded successfully.")

    # Run
    retriever = CustomMultiQueryRetriever(
        generated_queries=query_tag_list,
        retriever=vector_db.as_retriever(search_type="similarity_score_threshold",
                                         search_kwargs={'k': 20,
                                                        'score_threshold': 0.75}),
        llm_chain=RunnableLambda(lambda x: x + 1), include_original=True)
    # Results
    unique_docs = retriever.invoke(','.join(original_query_tags))

    if not unique_docs:
        print("No documents found.")
        return

    search_results = {
        "query": translated_query,
        "results": [
            {
                "index": index,
                "file": doc.metadata["file_name"],
                "source": doc.metadata["source"],
                "article_file": doc.metadata["article_file"],
                "article_tags": doc.page_content
            }
            for index, doc in enumerate(unique_docs)
        ]
    }

    pprint(search_results)


# if True:
#     handle_user_query("Da li je predsednik Republike Aleksandar Vučić govorio u Beogradu?", "1", "output")
#     #handle_user_query("Film Noir by Joseph Lewis", "1", "output")
#     #handle_user_query("Malawi Vice President Plane Crash", "1", "output")
#     exit(0)
# exit(0)

import argparse

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
