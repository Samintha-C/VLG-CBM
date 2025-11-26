# interventions/adapters/cbllm.py
from dataclasses import dataclass
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from loguru import logger
from datasets import load_dataset
from transformers import RobertaTokenizerFast, RobertaModel, GPT2TokenizerFast, GPT2Model

# Add CB-LLM classification path to sys.path
CBLLM_CLASSIFICATION_PATH = os.path.join(os.path.dirname(__file__), "../../../CB-LLMs/classification")
if CBLLM_CLASSIFICATION_PATH not in sys.path:
    sys.path.insert(0, CBLLM_CLASSIFICATION_PATH)

import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from utils import normalize, eos_pooling

@dataclass
class CBLLMRun:
    load_path: str  # e.g., "mpnet_acs/SetFit_sst2/roberta_cbm"
    sparse: bool = True  # Use sparse weights (W_g_sparse) if True
    backbone: str = "roberta"  # "roberta" or "gpt2"
    dataset: str = None  # Auto-inferred from load_path if None
    batch_size: int = 256
    max_length: int = 512
    device: str = "cuda"

def _parse_load_path(load_path: str):
    parts = [p for p in load_path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Invalid load_path format: {load_path}. Expected format: 'acs/dataset/backbone_cbm' or 'acs/dataset/backbone_cbm/cbl.pt'")
    
    acs = parts[0]  # e.g., "mpnet_acs"
    dataset_name = parts[1]  # e.g., "SetFit_sst2"
    
    if len(parts) >= 3:
        backbone = parts[2]  # e.g., "roberta_cbm"
        cbl_name = parts[-1] if len(parts) > 3 and parts[-1].endswith(".pt") else "cbl.pt"
    else:
        backbone = None
        cbl_name = "cbl.pt"
    
    if backbone:
        if "roberta" in backbone:
            backbone_type = "roberta"
        elif "gpt2" in backbone:
            backbone_type = "gpt2"
        else:
            raise ValueError(f"Unknown backbone in path: {backbone}")
    else:
        backbone_type = "roberta"
    
    dataset = dataset_name.replace('_', '/') if 'sst2' in dataset_name else dataset_name
    
    return acs, dataset, backbone_type, cbl_name

def _load_model(run: CBLLMRun, device):
    acs, dataset, backbone_type, cbl_name = _parse_load_path(run.load_path)
    
    if dataset not in CFG.concept_set:
        raise ValueError(f"Dataset {dataset} not found in CFG.concept_set. Available: {list(CFG.concept_set.keys())}")
    
    concept_set = CFG.concept_set[dataset]
    
    cbl_path = os.path.join(run.load_path, cbl_name)
    if not os.path.exists(cbl_path):
        raise FileNotFoundError(f"CBL model not found: {cbl_path}")
    
    if backbone_type == "roberta":
        if 'no_backbone' in cbl_name:
            logger.info("Loading CBL only (no backbone)...")
            cbl = CBL(len(concept_set), dropout=0.1).to(device)
            cbl.load_state_dict(torch.load(cbl_path, map_location=device))
            cbl.eval()
            preLM = RobertaModel.from_pretrained('roberta-base').to(device)
            preLM.eval()
            backbone_cbl = None
        else:
            logger.info("Loading backbone(roberta)+CBL...")
            backbone_cbl = RobertaCBL(len(concept_set), dropout=0.1).to(device)
            backbone_cbl.load_state_dict(torch.load(cbl_path, map_location=device))
            backbone_cbl.eval()
            preLM = None
            cbl = None
    elif backbone_type == "gpt2":
        if 'no_backbone' in cbl_name:
            logger.info("Loading CBL only (no backbone)...")
            cbl = CBL(len(concept_set), dropout=0.1).to(device)
            cbl.load_state_dict(torch.load(cbl_path, map_location=device))
            cbl.eval()
            preLM = GPT2Model.from_pretrained('gpt2').to(device)
            preLM.eval()
            backbone_cbl = None
        else:
            logger.info("Loading backbone(gpt2)+CBL...")
            backbone_cbl = GPT2CBL(len(concept_set), dropout=0.1).to(device)
            backbone_cbl.load_state_dict(torch.load(cbl_path, map_location=device))
            backbone_cbl.eval()
            preLM = None
            cbl = None
    else:
        raise ValueError(f"Unknown backbone: {backbone_type}")
    
    return preLM, cbl, backbone_cbl, concept_set, dataset, acs, cbl_name

def _extract_concept_features(run: CBLLMRun, split: str, preLM, cbl, backbone_cbl, device):
    acs, dataset, backbone_type, cbl_name = _parse_load_path(run.load_path)
    
    if backbone_type == "roberta":
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif backbone_type == "gpt2":
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f"Unknown backbone: {backbone_type}")
    
    logger.info(f"Loading {split} dataset...")
    if split == "train":
        dataset_split = load_dataset(dataset, split='train')
    elif split == "val":
        if dataset == 'SetFit/sst2':
            dataset_split = load_dataset(dataset, split='validation')
        else:
            raise ValueError(f"Dataset {dataset} doesn't have a validation split")
    elif split == "test":
        dataset_split = load_dataset(dataset, split='test')
    else:
        raise ValueError(f"Unknown split: {split}")
    
    logger.info(f"Tokenizing {split} dataset ({len(dataset_split)} samples)...")
    encoded_dataset = dataset_split.map(
        lambda e: tokenizer(e[CFG.example_name[dataset]], padding=True, truncation=True, max_length=run.max_length),
        batched=True, batch_size=len(dataset_split)
    )
    encoded_dataset = encoded_dataset.remove_columns([CFG.example_name[dataset]])
    if dataset == 'SetFit/sst2':
        encoded_dataset = encoded_dataset.remove_columns(['label_text'])
    if dataset == 'dbpedia_14':
        encoded_dataset = encoded_dataset.remove_columns(['title'])
    
    class ClassificationDataset(torch.utils.data.Dataset):
        def __init__(self, texts):
            self.texts = texts
        def __getitem__(self, idx):
            return {key: torch.tensor(values[idx]) for key, values in self.texts.items()}
        def __len__(self):
            return len(self.texts['input_ids'])
    
    loader = DataLoader(ClassificationDataset(encoded_dataset), batch_size=run.batch_size, shuffle=False)
    
    logger.info(f"Extracting concept features from {split}...")
    features_list = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if backbone_cbl is not None:
                batch_features = backbone_cbl(batch)
            else:
                text_features = preLM(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                if backbone_type == "roberta":
                    text_features = text_features[:, 0, :]
                elif backbone_type == "gpt2":
                    text_features = eos_pooling(text_features, batch["attention_mask"])
                batch_features = cbl(text_features)
            features_list.append(batch_features.cpu())
    
    X = torch.cat(features_list, dim=0)
    y = torch.LongTensor(encoded_dataset["label"])
    
    logger.info(f"Extracted {split} features: X shape {X.shape}, y shape {y.shape}")
    return X, y

def _load_split_tensors(run: CBLLMRun, split: str, device="cpu"):
    cache_path = os.path.join(run.load_path, f"{split}_concept_features.pt")
    cache_label_path = os.path.join(run.load_path, f"{split}_concept_labels.pt")
    
    if os.path.exists(cache_path) and os.path.exists(cache_label_path):
        logger.info(f"Loading cached {split} features from {cache_path}")
        X = torch.load(cache_path, map_location=device)
        y = torch.load(cache_label_path, map_location=device)
        logger.info(f"  Loaded X shape: {X.shape}, y shape: {y.shape}")
        return X, y
    
    logger.info(f"Cache not found, extracting {split} features from model...")
    preLM, cbl, backbone_cbl, concept_set, dataset, acs, cbl_name = _load_model(run, device)
    X, y = _extract_concept_features(run, split, preLM, cbl, backbone_cbl, device)
    
    logger.info(f"Caching {split} features to {cache_path}")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(X, cache_path)
    torch.save(y, cache_label_path)
    
    return X, y

def _normalize_features(X, run: CBLLMRun):
    acs, dataset, backbone_type, cbl_name = _parse_load_path(run.load_path)
    model_name = cbl_name[3:] if cbl_name.startswith("cbl") else cbl_name
    
    prefix = run.load_path
    if not os.path.isabs(prefix):
        prefix = os.path.join(os.getcwd(), prefix)
    
    train_mean_path = os.path.join(prefix, f'train_mean{model_name}')
    train_std_path = os.path.join(prefix, f'train_std{model_name}')
    
    if not os.path.exists(train_mean_path) or not os.path.exists(train_std_path):
        raise FileNotFoundError(f"Normalization stats not found: {train_mean_path} or {train_std_path}")
    
    train_mean = torch.load(train_mean_path)
    train_std = torch.load(train_std_path)
    
    X_norm, _, _ = normalize(X, d=0, mean=train_mean, std=train_std)
    X_norm = F.relu(X_norm)
    
    return X_norm

def get_loader(run: CBLLMRun, split: str, batch_size=256, num_workers=2, shuffle=False, device="cpu"):
    X, y = _load_split_tensors(run, split, device=device)
    X = _normalize_features(X, run)
    X = X.to(device)
    y = y.to(device)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=False)

def load_sparse_head(run: CBLLMRun, device="cuda"):
    acs, dataset, backbone_type, cbl_name = _parse_load_path(run.load_path)
    model_name = cbl_name[3:] if cbl_name.startswith("cbl") else cbl_name
    
    prefix = run.load_path
    if not os.path.isabs(prefix):
        prefix = os.path.join(os.getcwd(), prefix)
    
    W_g_path = os.path.join(prefix, "W_g")
    b_g_path = os.path.join(prefix, "b_g")
    if run.sparse:
        W_g_path += "_sparse"
        b_g_path += "_sparse"
    W_g_path += model_name
    b_g_path += model_name
    
    if not os.path.exists(W_g_path) or not os.path.exists(b_g_path):
        raise FileNotFoundError(f"Sparse head weights not found: {W_g_path} or {b_g_path}")
    
    logger.info(f"Loading sparse head: W from {os.path.basename(W_g_path)}, b from {os.path.basename(b_g_path)} to {device}")
    W = torch.load(W_g_path, map_location=device)
    b = torch.load(b_g_path, map_location=device)
    C, D = W.shape
    logger.info(f"  Loaded W shape: {W.shape}, b shape: {b.shape}, num_classes: {C}")
    return W, b, C

def forward_final(X, W, b):
    return X @ W.T + b

def predict(X, W, b):
    return forward_final(X, W, b).argmax(dim=1)

def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm

