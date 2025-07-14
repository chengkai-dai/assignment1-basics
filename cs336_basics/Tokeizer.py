from typing import Iterable
import regex

class Tokenizer:
    def __init__(self, 
                 vocab: dict[str, int], merges: dict[tuple[str, str], int], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.vocab_inverse = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merge_orders = {pair: i for i, pair in enumerate(merges)}

        self.special_tokens = special_tokens
        self.pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    @classmethod
    def from_file(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None):
        import json
        
        # Load vocabulary from JSON file
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        # Load merges from text file
        merges = {}
        with open(merges_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    parts = line.split()
                    if len(parts) >= 2:
                        # Create merge rule as tuple of first two tokens
                        merge_pair = (parts[0], parts[1])
                        # Use line number or index as priority (lower = higher priority)
                        merges[merge_pair] = len(merges)
        
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def _split_on_special_tokens(self, text: str) -> list[str]:
        """Split text on special tokens while preserving them"""
        if not self.special_tokens:
            return [text]
        
        # Create pattern to split on special tokens while preserving them
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = '|'.join(regex.escape(token) for token in sorted_special_tokens)
        return regex.split(f'({special_pattern})', text)
    
    def _get_pairs(self, word):
        """Get all adjacent pairs in a word"""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return pairs
    
    def _apply_bpe(self, word: list[bytes]) -> list[bytes]:
        """Apply BPE merges to a word"""
        if len(word) <= 1:
            return word
        while True:
            pairs = self._get_pairs(word)
            if not pairs:
                break
            
            # Find the merge with the highest priority (lowest index)
            best_pair = None
            best_priority = float('inf')
            
            for pair in pairs:
                if pair in self.merge_orders:
                    priority = self.merge_orders[pair]
                    if priority < best_priority:
                        best_pair = pair
                        best_priority = priority
            
            if best_pair is None:
                break
            
            # Apply the merge
            new_word = []
            i = 0
            while i < len(word):
                if (i < len(word) - 1 and 
                    word[i] == best_pair[0] and 
                    word[i + 1] == best_pair[1]):
                    # Merge the pair
                    merged = word[i] + word[i + 1]
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
        return word
        
    def encode(self, text: str) -> list[int]:
        parts = self._split_on_special_tokens(text)            
        token_ids = []
        
        for part in parts:
            if not part:
                continue
            
            # Check if this part is a special token
            if self.special_tokens and part in self.special_tokens:
                special_token_bytes = part.encode('utf-8')
                if special_token_bytes in self.vocab_inverse:
                    token_ids.append(self.vocab_inverse[special_token_bytes])
                continue
            
            tokens = regex.findall(self.pattern, part)
            
            for token in tokens:
                if not token:
                    continue
                
                # Convert to bytes and then to individual byte tokens
                token_bytes = token.encode('utf-8')
                
                # Start with individual bytes
                word = [bytes([b]) for b in token_bytes]

                # Apply BPE merges
                word = self._apply_bpe(word)
                
                # Convert to token IDs
                for token_bytes in word:
                    if token_bytes in self.vocab_inverse:
                        token_ids.append(self.vocab_inverse[token_bytes])
                    else:
                        # This shouldn't happen if vocab is complete
                        raise ValueError(f"Unknown token: {token_bytes}")
            
        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            if isinstance(text, str):
                yield from self.encode(text)
            else:
                # Handle file-like objects
                yield from self.encode(text.read() if hasattr(text, 'read') else str(text))
    
    def decode(self, ids: list[int]) -> str:
        if not ids:
                return ""
            
        # Convert token IDs to bytes
        byte_chunks = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_chunks.append(self.vocab[token_id])
            else:
                raise ValueError(f"Unknown token ID: {token_id}")
        
        # Concatenate all bytes
        all_bytes = b''.join(byte_chunks)
        
        # Decode to string, handling potential decoding errors
        try:
            return all_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Handle incomplete UTF-8 sequences by using error handling
            return all_bytes.decode('utf-8', errors='replace')
