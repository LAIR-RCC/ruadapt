from transformers import LlamaTokenizer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()


    llama_tokenizer = LlamaTokenizer(args.model_path, sp_model_kwargs={'model_file': args.model_path}, legacy=True)
    llama_tokenizer.init_kwargs['sp_model_kwargs'] = {}
    llama_tokenizer.save_pretrained(args.output_path)

