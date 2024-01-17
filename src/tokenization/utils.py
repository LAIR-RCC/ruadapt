from torch import nn
from tqdm import tqdm
import torch

def special_encode(token_str, tokenizer):
    shift = len(tokenizer.encode('家', add_special_tokens=False))
    tokens = tokenizer.encode('家' + token_str, add_special_tokens=False)
    return tokens[shift:]

def get_mean_vec(token, tokenizer, embeddings):
    tokens = special_encode(token, tokenizer)
    vector = embeddings[tokens].mean(axis=0)
    return vector

def reinit_embeddings_with_head_llama(model, tokenizer_old, tokenizer_new, mode='random', lm_head_init='tie'):
    assert lm_head_init in ['tie', 'hm']
    assert mode in ['random', 'mean']
    assert model.lm_head.bias is None

    vocab_size = len(tokenizer_new.get_vocab())
    model.config.vocab_size = vocab_size

    embeddings_old = model.model.embed_tokens.weight.data.clone()
    lm_head_old = model.lm_head.weight.data.clone()
    
    model.model.embed_tokens = nn.Embedding(model.config.vocab_size, model.config.hidden_size)
    model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False)
    if mode == 'random':
        model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    elif mode == 'mean':
        model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

        spec_tokens = set(tokenizer_new.special_tokens_map['additional_special_tokens']) if 'additional_special_tokens' in tokenizer_new.special_tokens_map else set()
        if 'eos_token' in tokenizer_new.special_tokens_map:
            spec_tokens.add(tokenizer_new.special_tokens_map['eos_token'])
        if 'bos_token' in tokenizer_new.special_tokens_map:
            spec_tokens.add(tokenizer_new.special_tokens_map['bos_token'])
        if 'unk_token' in tokenizer_new.special_tokens_map:
            spec_tokens.add(tokenizer_new.special_tokens_map['unk_token'])
        if 'pad_token' in tokenizer_new.special_tokens_map:
            spec_tokens.add(tokenizer_new.special_tokens_map['pad_token'])

        for i in tqdm(range(vocab_size)):
            token = tokenizer_new._tokenizer.id_to_token(i)
            if token in spec_tokens:
                continue
            
            vec = get_mean_vec(token, tokenizer_old, embeddings_old)
            model.model.embed_tokens.weight.data[i] = vec

            if lm_head_init == 'hm': # lm head mean
                vec = get_mean_vec(token, tokenizer_old, lm_head_old)
                model.lm_head.weight.data[i] = vec
            
        if lm_head_init == 'tie':
            model._tie_or_clone_weights(model.get_output_embeddings(), model.get_input_embeddings())
    else:
        raise Exception('NotImplemented')
    
def reinit_embeddings_with_head_llama_w2v(model, tokenizer_old, tokenizer_new, w2v_model):
    assert model.lm_head.bias is None

    vocab_size = len(tokenizer_new.get_vocab())
    model.config.vocab_size = vocab_size

    embeddings_old = model.model.embed_tokens.weight.data.clone()

    model.model.embed_tokens = nn.Embedding(model.config.vocab_size, model.config.hidden_size)
    model.lm_head = nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False)

    model.model.embed_tokens.weight.data.normal_(mean=0.0, std=model.config.initializer_range)
    model.lm_head.weight.data.normal_(mean=0.0, std=model.config.initializer_range)

    spec_tokens = set(tokenizer_new.special_tokens_map['additional_special_tokens']) if 'additional_special_tokens' in tokenizer_new.special_tokens_map else set()
    if 'eos_token' in tokenizer_new.special_tokens_map:
        spec_tokens.add(tokenizer_new.special_tokens_map['eos_token'])
    if 'bos_token' in tokenizer_new.special_tokens_map:
        spec_tokens.add(tokenizer_new.special_tokens_map['bos_token'])
    if 'unk_token' in tokenizer_new.special_tokens_map:
        spec_tokens.add(tokenizer_new.special_tokens_map['unk_token'])
    if 'pad_token' in tokenizer_new.special_tokens_map:
        spec_tokens.add(tokenizer_new.special_tokens_map['pad_token'])

    with torch.no_grad():
        for i in tqdm(range(vocab_size)):
            token = tokenizer_new._tokenizer.id_to_token(i)
            if token in spec_tokens:
                old_id = tokenizer_old._tokenizer.token_to_id(token)
                model.model.embed_tokens.weight.data[i] = embeddings_old[old_id]
                continue
            if i not in w2v_model:
                continue
            vec = torch.tensor(w2v_model[i], dtype=torch.float16)
            model.model.embed_tokens.weight.data[i] = vec

    model._tie_or_clone_weights(model.get_output_embeddings(), model.get_input_embeddings())