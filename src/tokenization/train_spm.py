import sentencepiece as spm
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files_dir')
    parser.add_argument('--model_name')
    parser.add_argument('--model_type', default='unigram')
    parser.add_argument('--vocab_size', default=32000, type=int)
    parser.add_argument('--count', default=100000, type=int)
    args = parser.parse_args()


    files = [os.path.join(args.input_files_dir, f) for f in os.listdir(args.input_files_dir)][:args.count]
    spm.SentencePieceTrainer.train(input=files, model_prefix=args.model_name, model_type=args.model_type, vocab_size=args.vocab_size, num_threads=30, split_digits=True, byte_fallback=True, train_extremely_large_corpus=True)

