import json
import sentencepiece as spm
import os
import re



spm.SentencePieceTrainer.train(
    input="времфайл",
    model_prefix='sentence_piece_tok',
    vocab_size=512,
    model_type='bpe',
    normalization_rule_name='identity'
    #character_coverage=0.99
)



#os.remove("времфайл")
