# This file will be called once the setup is complete. 
# It should perform all the preprocessing steps that are necessary to run the project. 
# All important arguments will be passed via the command line.
# The input files will adhere to the format specified in datastructure/input-file.json

import json
from os.path import join, split as split_path

from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import PromptTemplate

from util import constants


# TODO Implement the preprocessing steps here
def handle_input_file(file_location, output_path):
    empty_prompt = constants.OLLAMA_PROMPT

    with open(file_location) as f:
        data = json.load(f)

    instruction = "You are a helpful assistant working at the EU. It is your job to give users unbiased article recommendations. To do so, you always provide a list of tags, whenever you are prompted with an article. The tags should represent the core ideas of the article, and always be unbiased and in English. Respond with the tags only, separated by commas in a string array. Each tag can be 1 or more words, if needed"

    prompt = PromptTemplate.from_template(empty_prompt)
    llm = Ollama(model=constants.OLLAMA_MODEL_ID)
    chain = prompt | llm

    response = chain.invoke({"instruction": instruction, "input": "\n".join(data["content"])})

    tags = response.split(",")
    tags = list(set(map(lambda x: x.strip().lower(), tags)))

    file_name = split_path(file_location)[-1]

    #with open(join("data/transformed", file_name), "w") as f:
    with open(join(output_path, file_name), "w") as f:
        json.dump({
            "transformed_representation": tags
        }, f)

# if True:
#     handle_input_file("sample_data/article_1.json", "output")
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
