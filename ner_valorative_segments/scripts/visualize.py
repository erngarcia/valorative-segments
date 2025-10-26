import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_attention(text, model, tokenizer, max_len=50):
    model.eval()
    tokens = tokenizer.tokenize(text)[:max_len]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    with torch.no_grad():
        _, attention = model(torch.tensor([token_ids]))
    
    plt.figure(figsize=(12, 6))
    plt.imshow(attention.squeeze().cpu().numpy(), cmap='viridis')
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.colorbar()
    plt.show()

def plot_crf_transitions(model, label_vocab):
    """Visualize learned CRF transition matrix"""
    transitions = model.crf.transitions.detach().cpu().numpy()
    labels = [label_vocab[i] for i in range(len(label_vocab))]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(transitions, annot=True, fmt=".1f", 
                xticklabels=labels, yticklabels=labels)
    plt.title("CRF Transition Matrix")
    plt.show()
