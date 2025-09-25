#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Allowed deps only: torch, torch.nn, torch.optim, and stdlib (json, sys, os, re, datetime, collections)

import os, sys, json, re, time
from datetime import datetime, timezone
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# Utils: CLI parsing (very small)
# ----------------------------
def parse_args(argv):
    if len(argv) < 3:
        print("Usage: python train_embeddings.py <input_papers.json> <output_dir> [--epochs 50] [--batch_size 32]")
        sys.exit(1)
    input_path = argv[1]
    output_dir = argv[2]
    epochs = 50
    batch_size = 32
    i = 3
    while i < len(argv):
        if argv[i] == "--epochs" and i + 1 < len(argv) and argv[i+1].isdigit():
            epochs = int(argv[i+1]); i += 2
        elif argv[i] == "--batch_size" and i + 1 < len(argv) and argv[i+1].isdigit():
            batch_size = int(argv[i+1]); i += 2
        else:
            print(f"Unknown or malformed argument: {argv[i]}")
            sys.exit(1)
    return input_path, output_dir, epochs, batch_size

# ----------------------------
# Text cleaning
# ----------------------------
def clean_text(text):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    # keep letters and spaces only
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if len(w) >= 2]
    return words

# ----------------------------
# Data loading (HW#1 papers.json)
# ----------------------------
def load_abstracts(papers_json_path):
    print(f"Loading abstracts from {papers_json_path}...")
    with open(papers_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    abstracts = []
    paper_ids = []
    for i, p in enumerate(data):
        abs_txt = p.get("abstract", "")
        aid = p.get("arxiv_id") or p.get("id") or f"paper_{i}"
        if isinstance(abs_txt, str) and abs_txt.strip():
            abstracts.append(abs_txt)
            paper_ids.append(str(aid))
    print(f"Found {len(abstracts)} abstracts")
    return paper_ids, abstracts

# ----------------------------
# Vocab building: top-K (K=5000), idx 0 = <UNK>
# ----------------------------
def build_vocab(abstracts, top_k=5000):
    print("Building vocabulary...")
    total_tokens = 0
    counter = Counter()
    for abs_txt in abstracts:
        toks = clean_text(abs_txt)
        total_tokens += len(toks)
        counter.update(toks)
    most_common = counter.most_common(top_k)
    vocab_to_idx = {"<UNK>": 0}
    for i, (w, _) in enumerate(most_common, start=1):
        vocab_to_idx[w] = i
    idx_to_vocab = {str(i): w for w, i in vocab_to_idx.items()}
    vocab_size = len(vocab_to_idx)
    print(f"Vocabulary size: {vocab_size} words (top {top_k}), total_words: {total_tokens}")
    return vocab_to_idx, idx_to_vocab, vocab_size, total_tokens, counter

# ----------------------------
# Sequence encoding + BOW
# ----------------------------
def encode_sequences_to_bow(abstracts, vocab_to_idx, max_len=150):
    # Return a list of multi-hot vectors (0/1) as torch tensors
    bow_list = []
    for abs_txt in abstracts:
        toks = clean_text(abs_txt)[:max_len]  # pad/truncate step (we only use first max_len tokens)
        idxs = [vocab_to_idx.get(t, 0) for t in toks]
        # multi-hot BOW over vocabulary (presence)
        vec = torch.zeros(len(vocab_to_idx), dtype=torch.float32)
        for ix in idxs:
            if ix != 0:
                vec[ix] = 1.0
        bow_list.append(vec)
    return bow_list

# ----------------------------
# Model
# ----------------------------
class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)
        return rec, z

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

# ----------------------------
# Train
# ----------------------------
def train(model, bows, epochs=50, batch_size=32, lr=1e-3, device="cpu"):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    N = len(bows)
    bows_tensor = torch.stack(bows)  # [N, V]
    # simple train loop with shuffling
    for ep in range(1, epochs+1):
        perm = torch.randperm(N)
        total_loss = 0.0
        model.train()
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            batch = bows_tensor[idx].to(device)
            optimizer.zero_grad()
            rec, _ = model(batch)
            loss = criterion(rec, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg = total_loss / N if N > 0 else 0.0
        print(f"Epoch {ep}/{epochs}, Loss: {avg:.4f}")
    return avg  # final loss

# ----------------------------
# Inference: embeddings + per-sample recon loss
# ----------------------------
def compute_embeddings_and_losses(model, bows, device="cpu"):
    model.eval()
    criterion = nn.BCELoss(reduction="none")
    with torch.no_grad():
        bows_tensor = torch.stack(bows).to(device)
        rec, z = model(bows_tensor)
        # per-sample mean BCE
        losses = criterion(rec, bows_tensor).mean(dim=1).cpu().tolist()
        embeddings = z.cpu().tolist()
    return embeddings, losses

# ----------------------------
# Main
# ----------------------------
def main():
    input_path, output_dir, epochs, batch_size = parse_args(sys.argv)

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    paper_ids, abstracts = load_abstracts(input_path)
    if len(abstracts) == 0:
        print("No abstracts found in input file.")
        sys.exit(1)

    # Build vocab (top 5000)
    vocab_to_idx, idx_to_vocab, vocab_size, total_words, freq_counter = build_vocab(abstracts, top_k=5000)

    # Encode to BOW (multi-hot), with pad/truncate step respected by slicing
    bows = encode_sequences_to_bow(abstracts, vocab_to_idx, max_len=150)

    # Choose dims (must stay under 2,000,000 params with vocab_size up to 5000)
    hidden_dim = 128
    embedding_dim = 64

    # Create model
    model = TextAutoencoder(vocab_size=vocab_size, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    total_params = count_params(model)
    arch_str = f"{vocab_size} → {hidden_dim} → {embedding_dim} → {hidden_dim} → {vocab_size}"
    print(f"Model architecture: {arch_str}")
    print(f"Total parameters: {total_params:,}")

    # Verify parameter budget
    LIMIT = 2_000_000
    if total_params > LIMIT:
        print(f"ERROR: Parameter count {total_params:,} exceeds limit of {LIMIT:,}. Please reduce dims.")
        sys.exit(1)
    else:
        print(f"(under {LIMIT:,} limit ✓)")

    # Train
    print("\nTraining autoencoder...")
    start_ts = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    t0 = time.time()
    final_loss = train(model, bows, epochs=epochs, batch_size=batch_size, lr=1e-3, device="cpu")
    dur = time.time() - t0
    end_ts = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
    print(f"Training complete in {dur:.1f} seconds")

    # Save model
    model_path = os.path.join(output_dir, "model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_to_idx": vocab_to_idx,
        "model_config": {
            "vocab_size": vocab_size,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim
        }
    }, model_path)
    print(f"Saved model to {model_path}")

    # Inference for embeddings
    embs, rec_losses = compute_embeddings_and_losses(model, bows, device="cpu")

    # Save embeddings.json
    emb_out = []
    for pid, e, L in zip(paper_ids, embs, rec_losses):
        emb_out.append({
            "arxiv_id": pid,
            "embedding": [float(v) for v in e],
            "reconstruction_loss": float(L)
        })
    emb_path = os.path.join(output_dir, "embeddings.json")
    with open(emb_path, "w", encoding="utf-8") as f:
        json.dump(emb_out, f, ensure_ascii=False, indent=2)
    print(f"Saved embeddings to {emb_path}")

    # Save vocabulary.json
    vocab_path = os.path.join(output_dir, "vocabulary.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({
            "vocab_to_idx": vocab_to_idx,
            "idx_to_vocab": idx_to_vocab,
            "vocab_size": vocab_size,
            "total_words": total_words
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to {vocab_path}")

    # Save training_log.json
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "start_time": start_ts,
            "end_time": end_ts,
            "epochs": epochs,
            "final_loss": float(final_loss),
            "total_parameters": int(total_params),
            "papers_processed": len(abstracts),
            "embedding_dimension": embedding_dim
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved training log to {log_path}")

if __name__ == "__main__":
    main()
