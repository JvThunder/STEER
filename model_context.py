import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataclasses import dataclass, field

from model_configs import ModelConfig, get_config
from dataset_loaders import load_jbb_behaviors


@dataclass
class ModelContext:
    model_name: str
    model: object
    tokenizer: object
    device: str
    best_layer: int
    refusal_direction: object  # torch.Tensor
    config: ModelConfig = field(default_factory=ModelConfig)


def load_model(model_name: str):
    print("Loading model...")
    cfg = get_config(model_name)
    trc = cfg.trust_remote_code

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trc)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=trc,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            trust_remote_code=trc,
        )

    model.eval()
    device = next(model.parameters()).device.type
    print(f"Model loaded (device_map=auto, first param on: {device})\n")
    print(f"Using model: {model_name}  [{cfg.__class__.__name__}]")
    return model, tokenizer


def init_context(model_name: str, model, tokenizer, best_layer: int) -> ModelContext:
    """Build a ModelContext and compute the refusal direction."""
    device = next(model.parameters()).device.type
    cfg = get_config(model_name)

    ctx = ModelContext(
        model_name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=device,
        best_layer=best_layer,
        refusal_direction=None,
        config=cfg,
    )

    def _get_hidden(prompt):
        messages = [{"role": "user", "content": prompt}]
        formatted = cfg.apply_template(tokenizer, messages,
                                       add_generation_prompt=True, thinking=False)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[best_layer][0, -1, :].float()

    print("Computing refusal direction...")
    harmful_prompts = [b["goal"] for b in load_jbb_behaviors(split="harmful")]
    benign_prompts  = [b["goal"] for b in load_jbb_behaviors(split="benign")]
    harmful_acts = [_get_hidden(p) for p in harmful_prompts]
    benign_acts  = [_get_hidden(p) for p in benign_prompts]
    harmful_mean = torch.stack(harmful_acts).mean(dim=0)
    benign_mean  = torch.stack(benign_acts).mean(dim=0)
    ctx.refusal_direction = F.normalize(harmful_mean - benign_mean, p=2, dim=0)
    print("Done!\n")
    return ctx
