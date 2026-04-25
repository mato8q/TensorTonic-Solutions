import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        #PAD=0, UNK=1, BOS=2, EOS=3
        for i, token in enumerate(["<PAD>", "<UNK>", "<BOS>", "<EOS>"]):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        
        unique_words = set()
        for text in texts: 
        #ก็แค่หยิบทีละกล่อง แล้ว text คือของที่อยู่ในกล่องนั้น
            for word in text.lower().split():
                unique_words.add(word)

        for i, word in enumerate(sorted(unique_words), start=4):
          self.word_to_id[word] = i
          self.id_to_word[i] = word
          self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        ids = [] 
        for word in text.lower().split():  
            token_id = self.word_to_id.get(word, self.word_to_id[self.unk_token]) #.get(key, default) คือดึงค่าจาก dict ถ้าไม่เจอให้ใช้ค่า default แทน 
            ids.append(token_id)
            
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
        for token_id in ids:
            words.append(self.id_to_word.get(token_id, self.unk_token))
        
        return " ".join(words)