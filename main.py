from util import constants

if __name__ == '__main__':

    # # Load model directly
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # import torch
    #
    # mps_device = torch.device("mps")
    #
    # tokenizer = AutoTokenizer.from_pretrained("LeonKogler/MergedModelAllArticles")
    # model = AutoModelForCausalLM.from_pretrained("LeonKogler/MergedModelAllArticles", {"device_map": mps_device})
    #
    # # model = AutoPeftModelForCausalLM.from_pretrained(
    # #     constants.LEON_MODEL_ID,  # YOUR MODEL YOU USED FOR TRAINING
    # #     load_in_4bit=True,
    # # )
    # #tokenizer = AutoTokenizer.from_pretrained(constants.LEON_MODEL_ID)
    #
    # # Save to q4_k_m GGUF
    # model.save_pretrained_gguf("MergedModelAllArticlesTemp", tokenizer, quantization_method="q4_k_m")
    # model.push_to_hub_gguf("hf/MergedModelAllArticlesTemp", tokenizer, quantization_method="q4_k_m", token="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")

    exit(0)