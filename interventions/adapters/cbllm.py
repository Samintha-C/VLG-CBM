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
# Try multiple possible locations
possible_paths = [
    os.path.join(os.path.dirname(__file__), "../../../CB-LLMs/classification"),
    "/sc-cbint-vol/CB-LLMs/classification",
    "/sc-cbint-vol/twml/CB-LLMs/classification",
    os.path.join(os.path.dirname(__file__), "../../../../CB-LLMs/classification"),
]

CBLLM_CLASSIFICATION_PATH = None
for path in possible_paths:
    abs_path = os.path.abspath(path)
    if os.path.exists(abs_path) and os.path.exists(os.path.join(abs_path, "config.py")):
        CBLLM_CLASSIFICATION_PATH = abs_path
        break

if CBLLM_CLASSIFICATION_PATH is None:
    raise ImportError(
        f"Could not find CB-LLMs/classification directory. Tried: {possible_paths}. "
        "Please ensure CB-LLMs is cloned and accessible."
    )

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
    """
    Parse load_path to extract acs, dataset, backbone_type, and cbl_name.
    Handles both absolute and relative paths.
    Examples:
        - /sc-cbint-vol/cbllm-outputs/mpnet_acs/SetFit_sst2/roberta_cbm
        - mpnet_acs/SetFit_sst2/roberta_cbm
        - mpnet_acs/SetFit_sst2/roberta_cbm/cbl_acc.pt
    """
    # Normalize path and find the actual model directory
    load_path = os.path.normpath(load_path)
    if os.path.isfile(load_path):
        # If it's a file, get the directory
        model_dir = os.path.dirname(load_path)
        cbl_name = os.path.basename(load_path)
    else:
        # If it's a directory, find the cbl file
        model_dir = load_path
        cbl_files = [f for f in os.listdir(model_dir) if f.startswith("cbl") and f.endswith(".pt")]
        if cbl_files:
            cbl_name = cbl_files[0]  # Use first cbl file found
        else:
            cbl_name = "cbl_acc.pt"  # Default
    
    # Extract components from path
    parts = [p for p in model_dir.split(os.sep) if p]
    
    # Find mpnet_acs (or similar acs name) in the path
    acs_idx = None
    for i, part in enumerate(parts):
        if "acs" in part.lower() or part in ["mpnet_acs", "simcse_acs", "angle_acs"]:
            acs_idx = i
            break
    
    if acs_idx is None:
        raise ValueError(f"Could not find ACS directory (mpnet_acs, etc.) in path: {load_path}")
    
    acs = parts[acs_idx]
    
    # Next part should be dataset
    if acs_idx + 1 >= len(parts):
        raise ValueError(f"Invalid path structure. Expected: .../acs/dataset/backbone_cbm, got: {load_path}")
    
    dataset_name = parts[acs_idx + 1]
    
    # Next part should be backbone
    if acs_idx + 2 >= len(parts):
        raise ValueError(f"Invalid path structure. Expected: .../acs/dataset/backbone_cbm, got: {load_path}")
    
    backbone = parts[acs_idx + 2]
    
    # Determine backbone type
    if "roberta" in backbone:
        backbone_type = "roberta"
    elif "gpt2" in backbone:
        backbone_type = "gpt2"
    else:
        raise ValueError(f"Unknown backbone in path: {backbone}. Expected 'roberta' or 'gpt2' in directory name.")
    
    # Convert dataset name (SetFit_sst2 -> SetFit/sst2)
    dataset = dataset_name.replace('_', '/') if 'sst2' in dataset_name else dataset_name
    
    return acs, dataset, backbone_type, cbl_name

def _load_model(run: CBLLMRun, device):
    acs, dataset, backbone_type, cbl_name = _parse_load_path(run.load_path)
    
    if dataset not in CFG.concept_set:
        raise ValueError(f"Dataset {dataset} not found in CFG.concept_set. Available: {list(CFG.concept_set.keys())}")
    
    concept_set = CFG.concept_set[dataset]
    
    # Get the model directory
    model_dir = os.path.normpath(run.load_path)
    if os.path.isfile(model_dir):
        model_dir = os.path.dirname(model_dir)
    
    cbl_path = os.path.join(model_dir, cbl_name)
    if not os.path.exists(cbl_path):
        # Try to find any cbl file
        cbl_files = [f for f in os.listdir(model_dir) if f.startswith("cbl") and f.endswith(".pt")]
        if cbl_files:
            cbl_path = os.path.join(model_dir, cbl_files[0])
            logger.info(f"Using CBL file: {cbl_files[0]}")
        else:
            raise FileNotFoundError(f"CBL model not found in {model_dir}. Expected file like 'cbl_acc.pt'")
    
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
    # Extract model suffix (e.g., "_acc" from "cbl_acc.pt" or "_acc.pt" from "cbl_acc.pt")
    if cbl_name.startswith("cbl"):
        model_name = cbl_name[3:]  # Remove "cbl" prefix
        if model_name.startswith("_"):
            model_name = model_name[1:]  # Remove leading underscore if present
        if model_name.endswith(".pt"):
            model_name = model_name[:-3]  # Remove .pt extension
        if model_name:
            model_name = "_" + model_name  # Add back underscore for file naming
        else:
            model_name = ""
    else:
        model_name = ""
    
    prefix = os.path.normpath(run.load_path)
    if os.path.isfile(prefix):
        prefix = os.path.dirname(prefix)
    
    train_mean_path = os.path.join(prefix, f'train_mean{model_name}.pt')
    train_std_path = os.path.join(prefix, f'train_std{model_name}.pt')
    
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
    # Extract model suffix (e.g., "_acc" from "cbl_acc.pt")
    if cbl_name.startswith("cbl"):
        model_name = cbl_name[3:]  # Remove "cbl" prefix
        if model_name.startswith("_"):
            model_name = model_name[1:]  # Remove leading underscore if present
        if model_name.endswith(".pt"):
            model_name = model_name[:-3]  # Remove .pt extension
        if model_name:
            model_name = "_" + model_name  # Add back underscore for file naming
        else:
            model_name = ""
    else:
        model_name = ""
    
    prefix = os.path.normpath(run.load_path)
    if os.path.isfile(prefix):
        prefix = os.path.dirname(prefix)
    
    W_g_path = os.path.join(prefix, f"W_g{'_sparse' if run.sparse else ''}{model_name}.pt")
    b_g_path = os.path.join(prefix, f"b_g{'_sparse' if run.sparse else ''}{model_name}.pt")
    
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

