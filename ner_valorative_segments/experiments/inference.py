import torch
from typing import Dict, List, Tuple


def annotate_examples(model, text: str, word_vocab: Dict[str, int],
                      label_vocab: Dict[str, int], device: torch.device) -> List[Tuple[str, str]]:
    """
    Run model inference on a text and return token-label pairs.
    """
    model.eval()
    tokens = text.split()
    unk_id = word_vocab.get("<UNK>", 0)
    token_ids = [word_vocab.get(token, unk_id) for token in tokens]
    token_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        if hasattr(model, "crf"):
            predictions = model(token_tensor)[0]  # CRF returns list of lists
        else:
            output = model(token_tensor)
            logits = output[0] if isinstance(output, tuple) else output
            predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    idx_to_label = {idx: label for label, idx in label_vocab.items()}
    predicted_labels = [idx_to_label.get(idx, "O") for idx in predictions]

    return list(zip(tokens, predicted_labels))
