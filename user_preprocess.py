# This file will be called once the setup is complete. 
# It should perform all the preprocessing steps that are necessary to run the project. 
# All important arguments will be passed via the command line.
# The input files will adhere to the format specified in datastructure/input-file.json

import json
import os
from os.path import join, split as split_path

from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate

from util import constants
from util.library import util_common
from util.library.util_common import nanoid


# TODO Implement the preprocessing steps here
def handle_input_file(file_location, output_path):
    unique_id = nanoid(8)
    empty_prompt = constants.OLLAMA_PROMPT

    with open(file_location) as f:
        data = json.load(f)

    instruction = """You are a helpful assistant working at the EU. It is your job to give users unbiased article 
        recommendations. To do so, you always provide a list of tags, whenever you are prompted with an article. 
        These tags should represent the core ideas of the article, and always be unbiased and in English. 
        However, some of the generated tags may be non-English only if they are proper nouns like names or places. Each 
        tag may be 1 word or a phrase of maximum 8 words. Respond with only the tags, separated by new line characters. 
        Do not describe the article in any detail in the response, regardless of the language. Always respond with just the 
        tags. The first line should be be the header: Tags"""

    prompt = PromptTemplate.from_template(empty_prompt)
    llm = Ollama(model=constants.OLLAMA_MODEL_ID)
    chain = prompt | llm

    response = chain.invoke({"instruction": instruction, "input": "\n".join(data["content"])})
    raw_llm_output = response

    tags = response.split("\n")
    # Remove empty strings
    tags = list(filter(None, tags))
    tags = tags[1:]  # Remove the first line which is the identifier
    tags = list(set(map(lambda x: x.strip(), tags)))

    # Extract the file name and extension
    file_name, file_extension = os.path.splitext(os.path.basename(file_location))

    # Create the new file name by appending the unique ID
    new_file_name = f"{file_name}_{unique_id}{file_extension}"

    os.makedirs(output_path, exist_ok=True)
    with open(join(output_path, f"{file_name}{file_extension}"), "w") as f:
        json.dump({
            "transformed_representation": tags
        }, f)

    os.makedirs(constants.PRE_PROCESSED_PATH, exist_ok=True)
    with open(join(constants.PRE_PROCESSED_PATH, new_file_name), "w") as f:
        json.dump({
            "transformed_representation": tags,
            "llm_output": raw_llm_output
        }, f)

    util_common.copy_file_with_id(file_location, "data/articles", unique_id)

# if True:
#     #handle_input_file("sample_data/article_1.json", "output")
#     handle_input_file("test_data/serbian/Telegraf301.json", "output")
#     handle_input_file("test_data/english/title_Malawis_vice_president__9_others_killed_in_p.json", "output")
#     exit(0)
# exit(0)

# This is a useful argparse-setup, you probably want to use in your project:
import argparse

parser = argparse.ArgumentParser(description='Preprocess the data.')
parser.add_argument('--input', type=str, help='Path to the input data.', required=True, action="append")
parser.add_argument('--output', type=str, help='Path to the output directory.', required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    files_inp = args.input
    files_out = args.output

    for file_location in files_inp:
        handle_input_file(file_location, files_out)
