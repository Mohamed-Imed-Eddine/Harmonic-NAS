import torch 
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import pandas as pd
import numpy as np
import gzip
import html
from functools import lru_cache
import ftfy
import regex as re


@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = "bpe_simple_vocab_16e6.txt.gz"):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

# Get the image features for a single image input
def process_image_clip(in_img):
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    preprocess = Compose([
    Resize(112, T.InterpolationMode.BICUBIC),
    CenterCrop(112),
    ToTensor()
    ])
    image = preprocess(Image.open(in_img).convert("RGB"))
    image_input = torch.tensor(np.stack(image))
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]
    return image_input


# Get the text features for a single text input
def process_text_clip(in_text, context_length, bpe_path): 
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)   
    text_token = tokenizer.encode(in_text)
    text_input = torch.zeros(context_length, dtype=torch.float)
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']
    tokens = [sot_token] + text_token[:75] + [eot_token]
    text_input[:len(tokens)] = torch.tensor(tokens)
    text_input = text_input
    return text_input

class HarmemeMemesDatasetPol(torch.utils.data.Dataset):
    """Uses jsonl data to preprocess and serve dictionary of multimodal tensors for model input."""
    def __init__(
        self,
        data_path,
        img_dir,
        dir_ROI, 
        dir_ENT,
        context_length,
        bpe_path,
        extend=False,
        split_flag=None,
    ):

        self.samples_frame = pd.read_json(
            data_path, lines=True
        )
        self.samples_frame = self.samples_frame.reset_index(
            drop=True
        )
        self.samples_frame.image = self.samples_frame.apply(
            lambda row: (img_dir + '/' + row.image), axis=1
        )

        self.ROI_samples = dir_ROI
        self.ENT_samples = dir_ENT

        self.context_length = context_length
        self.bpe_path = bpe_path

        self.extend = extend
        
    def __len__(self):
        return len(self.samples_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.samples_frame.loc[idx, "id"]
        img_file_name = self.samples_frame.loc[idx, "image"]
        
        image_clip_input = process_image_clip(self.samples_frame.loc[idx, "image"])
        #image_vgg_feature = self.ROI_samples[idx] 

        text_clip_input = process_text_clip(self.samples_frame.loc[idx, "text"], self.context_length, self.bpe_path)
        #text_drob_feature = self.ENT_samples[idx]

        if self.extend == False:
            if self.samples_frame.loc[idx, "labels"][0]=="not harmful":
                lab=0
            elif self.samples_frame.loc[idx, "labels"][0]=="somewhat harmful":
                lab=1  
            else:
                lab=2
        else:
            if self.samples_frame.loc[idx, "labels"][0]=="not harmful":
                lab=0

            elif self.samples_frame.loc[idx, "labels"][0]=="somewhat harmful":

                if self.samples_frame.loc[idx, "labels"][1]=="individual":
                    lab = 1
                elif self.samples_frame.loc[idx, "labels"][1]=="organization":
                    lab = 2
                elif self.samples_frame.loc[idx, "labels"][1]=="community":
                    lab = 3
                elif self.samples_frame.loc[idx, "labels"][1]=="society":
                    lab = 4
                else:
                    raise NotImplementedError
            
            elif self.samples_frame.loc[idx, "labels"][0]=="very harmful":

                if self.samples_frame.loc[idx, "labels"][1]=="individual":
                    lab = 5
                elif self.samples_frame.loc[idx, "labels"][1]=="organization":
                    lab = 6
                elif self.samples_frame.loc[idx, "labels"][1]=="community":
                    lab = 7
                elif self.samples_frame.loc[idx, "labels"][1]=="society":
                    lab = 8
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        label = torch.tensor(lab)

        sample = [image_clip_input, text_clip_input, label]
        
        return sample