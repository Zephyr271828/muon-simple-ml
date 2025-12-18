import pdb
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

class MoonDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=512):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        # self.texts = dataset["train"]["text"]
        self.texts = dataset["text"]
        self.max_length = max_length
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
    #     if os.path.exists(f"{self.dataset_name}.bin"):
    #         self.tokens = torch.load(f"{self.dataset_name}.bin")
    #     else:
        for text in tqdm(self.texts, desc="Tokenizing texts"):
            encoded = self.tokenizer.encode(text, add_special_tokens=True)
            self.tokens.extend(encoded)
        # torch.save(self.tokens, f"{self.dataset_name}.bin")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * (self.max_length)
        end_idx = start_idx + (self.max_length)
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data, torch.tensor([0])  # Dummy label for compatibility
    
class LMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, labels=None):
        del labels
        outputs = self.model(input_ids=inputs, labels=inputs)
        return type("Out", (), {"loss": outputs.loss, "logits": outputs.logits})
    
def get_model_and_dataloader(
    model_name="qwen", 
    dataset_name="openwebtext-100k", 
    hidden_size=1024, 
    limit=1_000_000_000
):
    # pdb.set_trace()
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }
    train_dataset = load_dataset(
        name2path[dataset_name], split="train", trust_remote_code=True
    )
    train_dataset = train_dataset.select(range(min(limit, len(train_dataset))))
    if model_name == "qwen":
        tokenizer = Qwen2Tokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", trust_remote_code=True
        )
    elif model_name == "llama":
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B", trust_remote_code=True
        )
    else:
        assert 0, f"model {model_name} not supported"
    train_dataset = MoonDataset(dataset_name, train_dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    if model_name == "qwen":
        config = Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=4864,
            max_position_embeddings=513,
            max_window_layers=12,
            model_type="qwen2",
            num_attention_heads=16,
            num_hidden_layers=12,
            num_key_value_heads=16,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=1024,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=False,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=151936,
        )
        model = Qwen2ForCausalLM(config)
    elif model_name == "llama":
        assert hidden_size == 2048, "Llama-3.1-8B requires hidden_size of 2048"
        config = LlamaConfig(
            attention_dropout=0.0,
            bos_token_id=128000,
            eos_token_id=128001,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=8192,
            max_position_embeddings=513,
            model_type="llama3",
            num_attention_heads=32,
            num_hidden_layers=16,
            num_key_value_heads=8,
            rms_norm_eps=1e-05,
            rope_theta=500000.0,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=False,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=128256,
        )
        model = LlamaForCausalLM(config)
    else:
        assert 0, f"model {model_name} not supported"
    return LMWrapper(model), train_loader