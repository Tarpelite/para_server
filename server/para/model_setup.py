from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from transformers import BertTokenizer, RobertaTokenizer
from ...unilm.s2s_ft.modeling_decoding import BertForSeq2SeqDecoder, BertConfig
from transformers.tokenization_bert import whitespace_tokenize
import ...unilm.s2s_ft.s2s_loader as seq2seq_loader
from ...unilm.s2s_ft.utils import load_and_cache_examples
from transformers import \
    BertTokenizer, RobertaTokenizer
from ...unilm.s2s_ft.tokenization_unilm import UnilmTokenizer
from ...unilm.s2s_ft.tokenization_minilm import MinilmTokenizer

import json

class WhitespaceTokenizer(object):
    def tokenize(self, text):
        return whitespace_tokenize(text)

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list

class unilm_paraphrase_generator:
    def __init__(self, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            generator_config = json.load(f)

        self.tokenizer = BertTokenizer.from_pretrained(generator_config["tokenizer_name"])
        self.config = BertConfig.from_pretrained(generator_config["config_path"])
        self.max_seq_length = generator_config['max_seq_length']
        self.max_tgt_length = generator_config['max_tgt_length']
        self.beam_size = generator_config['beam_size']
        self.pos_shift = False
        self.batch_size = 1
        self.bi_uni_pipeline = []

        self.bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(
        list(vocab.keys()), self.tokenizer.convert_tokens_to_ids, self.max_seq_length,
        max_tgt_length=self.max_tgt_length, pos_shift=self.pos_shift,
        source_type_id=self.config.source_type_id, target_type_id=self.config.target_type_id, 
        cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token, pad_token=tokenizer.pad_token))
        self.mask_word_id, self.eos_word_ids, self.sos_word_id = self.tokenizer.convert_tokens_to_ids(
        [self.tokenizer.mask_token, self.tokenizer.sep_token, self.tokenizer.sep_token])
        self.forbid_ignore_set = None
        if generator_config['forbid_ignore_word']:
            w_list = []
            for w in generator_config['forbid_ignore_word'].split('|'):
                if w.startswith('[') and w.endswith(']'):
                    w_list.append(w.upper())
                else:
                    w_list.append(w)
            self.forbid_ignore_set = set(self.tokenizer.convert_tokens_to_ids(w_list))
        
        self.device = torch.device(
             "cuda" if torch.cuda.is_available() else "cpu"
        )

        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        if torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(123)
        self.model = BertForSeq2SeqDecoder.from_pretrained(
            generator_config['model_path'].strip(), config=self.config, mask_word_id=self.mask_word_id, search_beam_size=self.beam_size,
            length_penalty=False, eos_id=self.eos_word_ids, sos_id=self.sos_word_id,
            forbid_duplicate_ngrams=False, forbid_ignore_set=self.forbid_ignore_set,
            ngram_size=3, min_len=1, mode='s2s',
            max_position_embeddings=self.max_seq_length, pos_shift=self.pos_shift, 
        )

        self.model.to(self.device)

    def decode(self, input_text):
        torch.cuda.empty_cache()
        self.model.eval()
        next_i = 0
        max_src_length = self.max_seq_length - 2 - self.max_tgt_length
        source_tokens = self.tokenizer.tokenize(input_text)
        features = [
            {
                "source_ids": self.tokenizer.convert_tokens_to_ids(source_tokens),
                "target_ids":[]
            }
        ]
        to_pred = features
        
        input_lines = []
        for line in to_pred:
            input_lines.append(self.tokenizer.convert_ids_to_tokens(line["source_ids"])[:self.max_src_length])
        

        input_lines = sorted(list(enumerate(input_lines)),
                            key=lambda x: -len(x[1]))
        output_lines = [""] * len(input_lines)
        score_trace_list = [None] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / 1)

        with tqdm(total=total_batch) as pbar:
            batch_count = 0
            first_batch = True
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + self.batch_size]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += self.batch_size
                batch_count += 1
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    for proc in self.bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = seq2seq_loader.batch_list_to_batch_tensors(
                        instances)
                    batch = [
                        t.to(self.device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                    traces = self.model(input_ids, token_type_ids,
                                position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
                    if self.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces.tolist()
                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = self.tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in (self.tokenizer.sep_token, self.tokenizer.pad_token):
                                break
                            output_tokens.append(t)
                        if args.model_type == "roberta":
                            output_sequence = self.tokenizer.convert_tokens_to_string(output_tokens)
                        else:
                            output_sequence = ' '.join(detokenize(output_tokens))
                        if '\n' in output_sequence:
                            output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))
                        output_lines[buf_id[i]] = output_sequence
                        
                        
                pbar.update(1)
                first_batch = False
        
        return output_lines[0]


        




