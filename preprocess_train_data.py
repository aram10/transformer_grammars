import re
import sentencepiece as spm

from transformer_grammars.data import text_processing
from transformer_grammars.data.sp_utils import SentencePieceVocab


def preprocess_train_data(total_vocab_size: int = 32678, data_dir: str = "data"):
    for split in ["train", "valid", "test"]:
        text_processing.convert_to_choe_charniak(f"{data_dir}/{split}.txt", f"{data_dir}/{split}.choecharniak", has_preterms=True, untyped_closing_terminal=False)

    with open(f"{data_dir}/train.choecharniak", 'r', encoding='utf-8') as file:
        content = file.read()
    content = re.sub(r' ', '\n', content)
    NON_TERMINALS = ','.join([f"({x}" for x in sorted(set(re.findall(r'\((\S+)', content)))] + [f"{x})" for x in sorted(set(re.findall(r'(\S+)\)', content)))])

    spm.SentencePieceTrainer.train(f'--input={data_dir}/train.choecharniak \
                                     --model_prefix=spm/spm \
                                     --vocab_size={total_vocab_size} \
                                     --character_coverage=1.0 \
                                     --pad_id=0 \
                                     --bos_id=1 \
                                     --eos_id=2 \
                                     --unk_id=3 \
                                     --user_defined_symbols={NON_TERMINALS} \
                                     --max_sentence_length=100000 \
                                     --shuffle_input_sentence=true')

    sp = spm.SentencePieceProcessor(model_file='spm/spm.model')
    with open('spm/spm.vocab', 'r', encoding='utf-8') as f:
        contents = f.readlines()
    vocab = SentencePieceVocab.from_vocab_file(contents)
    for split in ["train", "valid", "test"]:
        with open(f'{data_dir}/{split}.choecharniak', 'r', encoding='utf-8') as rf, open(f'{data_dir}/{split}.csv', 'w', encoding='utf-8') as wf:
            for line in rf:
                token_ids = text_processing.postprocess_token_ids(sp.encode_as_ids(line), vocab)
                wf.write(",".join(str(x) for x in token_ids) + "\n")

