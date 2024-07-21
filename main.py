import json
import os
import subprocess
import sys
from os.path import join, isfile


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def transform_output(output):
    lines = output.split('\n')
    lines = map(lambda x: x.strip(), lines)
    lines = filter(lambda x: x != '', lines)
    return "◦◦◦ " + "\n◦◦◦ ".join(lines)


def import_test_data(language: str):
    for file_name in os.listdir("test_data/" + language):
        if file_name.endswith('.json'):
            args = ["--output", "output/delete", "--input", f"test_data/{language}/{file_name}"]
            result = subprocess.run([sys.executable, "user_preprocess.py", *args], check=False,
                                    capture_output=True)
            if result.returncode != 0:
                print(bcolors.FAIL + "> The preprocess script did not run successfully.")
                print(bcolors.FAIL + transform_output(result.stderr.decode()))


def init_vector_store():
    query = "What is the information on the Malawi Vice President Plane Crash?"
    query_id = "init_query"
    out_dir = "output/delete"
    args = ["--query", query, "--query_id", query_id, "--output", out_dir]

    result = subprocess.run([sys.executable, "user_inference.py", *args], check=False, capture_output=True)
    if result.returncode != 0:
        print(bcolors.FAIL + "> The inference script did not run successfully.")
        print(bcolors.FAIL + transform_output(result.stderr.decode()))

    else:
        try:
            assert isfile(join(out_dir, f"{query_id}.json")), f"The file {query_id}.json was not created."
            with open(join(out_dir, f"{query_id}.json"), "r") as f:
                filedata = json.load(f)
                assert "detected_language" in filedata, f"The key 'detected_language' was not found in the file {query_id}.json."
                assert "generated_queries" in filedata, f"The key 'generated_queries' was not found in the file {query_id}.json."

            print(bcolors.OKGREEN + "> The inference script ran successfully.")
            print(bcolors.OKBLUE + transform_output(result.stdout.decode()))

        except Exception as e:
            print(bcolors.FAIL + "> The inference script did not create the expected output.")
            print(bcolors.FAIL + transform_output(str(e)))


if __name__ == "__main__":
    import_test_data("english")
    import_test_data("serbian")

    init_vector_store()
