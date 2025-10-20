import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import os
import sys
from typing import Iterator, List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# tiktoken 사용
try:
    import tiktoken
except ImportError as e:
    print(f"tiktoken import 오류: {e}")
    print("필요한 패키지를 설치해주세요: conda install tiktoken")
    sys.exit(1)

# 하이퍼파라미터
BATCH_SIZE = 32
BLOCK_SIZE = 128  # 컨텍스트 길이
MAX_ITERS = 10000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
EVAL_ITERS = 200
VOCAB_SIZE = 10000
N_EMBD = 256  # 임베딩 차원
N_HEAD = 8    # 어텐션 헤드 수
N_LAYER = 6   # 레이어 수
DROPOUT = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'사용 디바이스: {DEVICE}')

class SimpleTokenizer:
    """간단한 토크나이저 (단어 기반)"""
    
    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>"]
        
    def train_tokenizer(self, text: str, save_path: str = "tokenizer.json"):
        """텍스트에서 토크나이저 훈련"""
        print("토크나이저 훈련 중...")
        
        # 텍스트를 단어로 분할
        words = text.split()
        
        # 단어 빈도 계산
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # 빈도순으로 정렬하여 어휘 생성
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 특수 토큰 추가
        self.word_to_idx = {}
        self.idx_to_word = {}
        
        # 특수 토큰 먼저 추가
        for i, token in enumerate(self.special_tokens):
            self.word_to_idx[token] = i
            self.idx_to_word[i] = token
        
        # 일반 단어 추가 (vocab_size 제한)
        vocab_count = len(self.special_tokens)
        for word, count in sorted_words:
            if vocab_count >= self.vocab_size:
                break
            if word not in self.word_to_idx:
                self.word_to_idx[word] = vocab_count
                self.idx_to_word[vocab_count] = word
                vocab_count += 1
        
        # 토크나이저 저장
        tokenizer_data = {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "vocab_size": len(self.word_to_idx)
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        
        print(f"토크나이저가 {save_path}에 저장되었습니다.")
        print(f"실제 어휘 크기: {len(self.word_to_idx)}")
        
        return len(self.word_to_idx)
    
    def load_tokenizer(self, path: str = "tokenizer.json"):
        """저장된 토크나이저 로드"""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                tokenizer_data = json.load(f)
            
            self.word_to_idx = tokenizer_data["word_to_idx"]
            self.idx_to_word = {int(k): v for k, v in tokenizer_data["idx_to_word"].items()}
            print(f"토크나이저가 {path}에서 로드되었습니다.")
            return len(self.word_to_idx)
        else:
            print(f"토크나이저 파일 {path}를 찾을 수 없습니다.")
            return None
    
    def encode(self, text: str) -> List[int]:
        """텍스트를 토큰으로 인코딩"""
        if not self.word_to_idx:
            raise ValueError("토크나이저가 훈련되지 않았습니다.")
        
        words = text.split()
        tokens = []
        for word in words:
            if word in self.word_to_idx:
                tokens.append(self.word_to_idx[word])
            else:
                tokens.append(self.word_to_idx["<|unk|>"])
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """토큰을 텍스트로 디코딩"""
        if not self.idx_to_word:
            raise ValueError("토크나이저가 훈련되지 않았습니다.")
        
        words = []
        for token in tokens:
            if token in self.idx_to_word:
                word = self.idx_to_word[token]
                if word not in self.special_tokens:
                    words.append(word)
        
        return " ".join(words)

class RNNAttention(nn.Module):
    """RNN 기반 언어 모델 (LSTM + 어텐션)"""
    
    def __init__(self, vocab_size: int, n_embd: int, n_hidden: int, n_layers: int, dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        
        # 임베딩 레이어
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, n_embd)
        
        # LSTM 레이어
        self.lstm = nn.LSTM(n_embd, n_hidden, n_layers, batch_first=True, dropout=dropout)
        
        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(n_hidden, num_heads=8, dropout=dropout, batch_first=True)
        
        # 출력 레이어
        self.ln_f = nn.LayerNorm(n_hidden)
        self.lm_head = nn.Linear(n_hidden, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 임베딩
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        x = self.dropout(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, n_hidden)
        
        # 어텐션
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # (B, T, n_hidden)
        
        # 잔차 연결
        x = lstm_out + attn_out
        
        # 출력
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """텍스트 생성"""
        for _ in range(max_new_tokens):
            # 컨텍스트 크롭
            idx_cond = idx[:, -BLOCK_SIZE:]
            # 예측
            logits, loss = self(idx_cond)
            # 마지막 토큰만 사용
            logits = logits[:, -1, :]
            # 확률 분포
            probs = F.softmax(logits, dim=-1)
            # 샘플링
            idx_next = torch.multinomial(probs, num_samples=1)
            # 시퀀스에 추가
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class TransformerLanguageModel(nn.Module):
    """Transformer 기반 언어 모델"""
    
    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, dropout: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        
        # 임베딩 레이어
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, n_embd)
        
        # Transformer 블록들
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        ])
        
        # 출력 레이어
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 임베딩
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        
        # Transformer 블록들
        for block in self.blocks:
            x = block(x)
        
        # 출력
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        """텍스트 생성"""
        for _ in range(max_new_tokens):
            # 컨텍스트 크롭
            idx_cond = idx[:, -BLOCK_SIZE:]
            # 예측
            logits, loss = self(idx_cond)
            # 마지막 토큰만 사용
            logits = logits[:, -1, :]
            # 확률 분포
            probs = F.softmax(logits, dim=-1)
            # 샘플링
            idx_next = torch.multinomial(probs, num_samples=1)
            # 시퀀스에 추가
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class TransformerBlock(nn.Module):
    """Transformer 블록"""
    
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    """멀티헤드 어텐션"""
    
    def __init__(self, num_heads: int, head_size: int, dropout: float = 0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class Head(nn.Module):
    """단일 어텐션 헤드"""
    
    def __init__(self, head_size: int, dropout: float = 0.2):
        super().__init__()
        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v
        return out

class FeedForward(nn.Module):
    """피드포워드 네트워크"""
    
    def __init__(self, n_embd: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        
    def forward(self, x):
        return self.net(x)

def get_batch(data, batch_size, block_size, device):
    """배치 데이터 생성"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, device):
    """손실 추정"""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, BATCH_SIZE, BLOCK_SIZE, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train_model(model, train_data, val_data, model_name, max_iters=MAX_ITERS):
    """모델 훈련"""
    print(f"\n{model_name} 모델 훈련 시작...")
    
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 훈련 기록
    train_losses = []
    val_losses = []
    iterations = []
    
    for iter in tqdm(range(max_iters), desc=f"{model_name} 훈련"):
        # 손실 평가
        if iter % EVAL_INTERVAL == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, EVAL_ITERS, DEVICE)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            train_losses.append(losses['train'].item())
            val_losses.append(losses['val'].item())
            iterations.append(iter)
        
        # 배치 샘플링
        xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        
        # 순전파, 역전파, 최적화
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # 모델 저장
    torch.save(model.state_dict(), f'{model_name.lower()}_model.pt')
    print(f"{model_name} 모델이 저장되었습니다.")
    
    return train_losses, val_losses, iterations

def generate_text(model, tokenizer, prompt="", max_new_tokens=200):
    """텍스트 생성"""
    model.eval()
    with torch.no_grad():
        if prompt:
            context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
        else:
            context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        
        generated = model.generate(context, max_new_tokens)
        generated_text = tokenizer.decode(generated[0].tolist())
        return generated_text

def plot_training_curves(rnn_losses, transformer_losses, iterations):
    """훈련 곡선 시각화"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(iterations, rnn_losses[0], label='RNN Train', color='blue')
    plt.plot(iterations, rnn_losses[1], label='RNN Val', color='blue', linestyle='--')
    plt.plot(iterations, transformer_losses[0], label='Transformer Train', color='red')
    plt.plot(iterations, transformer_losses[1], label='Transformer Val', color='red', linestyle='--')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Curves Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(iterations, rnn_losses[1], label='RNN Validation', color='blue')
    plt.plot(iterations, transformer_losses[1], label='Transformer Validation', color='red')
    plt.xlabel('Iterations')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 함수"""
    print("언어 모델 비교 실험 시작")
    
    # 데이터 로드 (Colab 호환)
    data_path = 'Char_Transformer_Language_Model/input.txt'
    if not os.path.exists(data_path):
        # Colab 환경에서 샘플 데이터 사용
        print("원본 데이터를 찾을 수 없습니다. 샘플 데이터를 사용합니다.")
        sample_text = """
        To be or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles,
        And by opposing end them. To die—to sleep,
        No more; and by a sleep to say we end
        The heart-ache and the thousand natural shocks
        That flesh is heir to: 'tis a consummation
        Devoutly to be wish'd. To die, to sleep;
        To sleep, perchance to dream—ay, there's the rub:
        For in that sleep of death what dreams may come,
        When we have shuffled off this mortal coil,
        Must give us pause—there's the respect
        That makes calamity of so long life.
        """ * 100
        text = sample_text
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
    
    print(f"데이터 크기: {len(text):,} 문자")
    
    # 토크나이저 설정
    tokenizer = SimpleTokenizer(VOCAB_SIZE)
    
    # 토크나이저 훈련 또는 로드
    tokenizer_path = "tokenizer.json"
    if os.path.exists(tokenizer_path):
        vocab_size = tokenizer.load_tokenizer(tokenizer_path)
    else:
        vocab_size = tokenizer.train_tokenizer(text, tokenizer_path)
    
    if vocab_size is None:
        print("토크나이저 로드 실패")
        return
    
    # 데이터 토큰화
    print("데이터 토큰화 중...")
    tokens = tokenizer.encode(text)
    print(f"토큰 수: {len(tokens):,}")
    
    # 훈련/검증 분할
    n = int(0.9 * len(tokens))
    train_data = torch.tensor(tokens[:n], dtype=torch.long)
    val_data = torch.tensor(tokens[n:], dtype=torch.long)
    
    print(f"훈련 데이터: {len(train_data):,} 토큰")
    print(f"검증 데이터: {len(val_data):,} 토큰")
    
    # 모델 생성
    rnn_model = RNNAttention(vocab_size, N_EMBD, N_EMBD, N_LAYER, DROPOUT)
    transformer_model = TransformerLanguageModel(vocab_size, N_EMBD, N_HEAD, N_LAYER, DROPOUT)
    
    print(f"\nRNN 모델 파라미터 수: {sum(p.numel() for p in rnn_model.parameters()) / 1e6:.2f}M")
    print(f"Transformer 모델 파라미터 수: {sum(p.numel() for p in transformer_model.parameters()) / 1e6:.2f}M")
    
    # 모델 훈련
    rnn_train_losses, rnn_val_losses, iterations = train_model(
        rnn_model, train_data, val_data, "RNN"
    )
    
    transformer_train_losses, transformer_val_losses, iterations = train_model(
        transformer_model, train_data, val_data, "Transformer"
    )
    
    # 훈련 곡선 시각화
    plot_training_curves(
        (rnn_train_losses, rnn_val_losses),
        (transformer_train_losses, transformer_val_losses),
        iterations
    )
    
    # 텍스트 생성 비교
    print("\n=== 텍스트 생성 비교 ===")
    
    prompt = "To be or not to be"
    print(f"\n프롬프트: '{prompt}'")
    
    print("\nRNN 모델 생성:")
    rnn_text = generate_text(rnn_model, tokenizer, prompt, 200)
    print(rnn_text)
    
    print("\nTransformer 모델 생성:")
    transformer_text = generate_text(transformer_model, tokenizer, prompt, 200)
    print(transformer_text)
    
    # 최종 성능 비교
    print("\n=== 최종 성능 비교 ===")
    print(f"RNN 최종 검증 손실: {rnn_val_losses[-1]:.4f}")
    print(f"Transformer 최종 검증 손실: {transformer_val_losses[-1]:.4f}")
    
    if transformer_val_losses[-1] < rnn_val_losses[-1]:
        print("Transformer 모델이 더 나은 성능을 보입니다.")
    else:
        print("RNN 모델이 더 나은 성능을 보입니다.")
    
    print("\n실험이 완료되었습니다!")

if __name__ == "__main__":
    main()
