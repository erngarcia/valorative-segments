import yaml
from experiments.run_experiment import run_experiment
import itertools
import json
import os
from preprocessing.dataset import NERDataset
import torch

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def annotate_examples(model, text, word_vocab, label_vocab, device):
    model.eval()

    # Tokenize and map to IDs
    tokens = text.split()
    unk_id = word_vocab.get("<UNK>", 0)
    token_ids = [word_vocab.get(token, unk_id) for token in tokens]

    # Convert to tensor
    token_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        if hasattr(model, "crf"):
            predictions = model(token_tensor)[0]  # CRF returns list of lists
        else:
            output = model(token_tensor)
            logits = output[0] if isinstance(output, tuple) else output
            predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    # Convert predicted label indices to strings
    idx_to_label = {idx: label for label, idx in label_vocab.items()}
    predicted_labels = [idx_to_label.get(idx, "O") for idx in predictions]

    return list(zip(tokens, predicted_labels))

def main():
    search_space = {
        "lr": [1e-3, 1e-4],
        "batch_size": [16, 32],
        "hidden_dim": [128, 192],
        "dropout": [0.3],
        }
    

    keys = list(search_space.keys())
    combinations = list(itertools.product(*search_space.values()))
    overall_best_f1 = 0
    overall_best_model_name = None
    overall_best_matrix = None
    overall_best_metrics = None
    best_hparams = None
    overall_errors = None
    best_model = None
    best_dataset = None
    best_vocab = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, combo in enumerate(combinations):
        hparams = dict(zip(keys, combo))
        best_f1, best_metrics, best_matrix, best_model_name, errors, model, dataset, word_vocab, label_vocab = run_experiment(hparams, experiment_number=i + 1)
        print("=" * 80)
        if best_f1 > overall_best_f1:
            overall_best_f1 = best_f1
            overall_best_model_name = best_model_name
            overall_best_matrix = best_matrix
            overall_best_metrics = best_metrics
            best_hparams = hparams
            overall_errors = errors
            best_model = model
            best_dataset = dataset
            best_vocab = (word_vocab, label_vocab)

    output_dir = "artifacts"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"errors_{overall_best_model_name}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(overall_errors, f, indent=4, ensure_ascii=False)


    print(f"\nSaved best model error report to {output_path}")
    word_vocab, label_vocab = best_vocab
    sample_sentences = [
                "cuándo han protegido a un hombre cuando una mujer lo golpea sabes lo que la gente dice es maricón por no defenderse no es un hombre eso son casos que siempre se van a dar no importando si eres hombre o mujer no excluyas a los hombres porque eso no es igualdad también hay hombres asesinados por sus mujeres que no reciben justicia mujeres como ustedes no quieren igualdad sino que quieren ser superior a los hombres",
                "de dónde sacaste ese caso de de cada niñas sufren algún tipo de abuso sexual dijiste que no los incriminaron por falta de pruebas en mi opinión eso no tiene tanto sentido ya que con los testimonios y las pruebas realizadas en españa es suficiente para meter a la cárcel a un sujeto",
                "tú abogado fuese pésimo y por eso perdieses la demanda ya que tenías todas las de ganar si lo que dices es verídico",
                "tonterías a parte ni a mí ni a cualquier persona con un poco de cabeza se le ocurriría culpar a los hombres de contraer y contagiar el coronavirus por el hecho de que mueran más hombres que mujeres al infectarse del mismo",
                "no pongas en mi boca ni en la boca de muchísimas otras mujeres que jamás hemos dicho comentarios tan deleznables como este",
                "comentarios tan deleznables como este ni mucho menos para desprestigiar el movimiento feminista utilizando esa excusa para decirnos que estamos mal de la cabeza y que somos unas enfermas",
                "no pero tú porque hayas escuchado disparates de una parte mínima del colectivo ya te crees con el derecho de llamarme a mí y a muchas otras mujeres enfermas",
                "viola los violadores y los hay hombres y los hay mujeres ellas menos y más en los casos de niños pq a la inversa pasan también",
                "en eeuu más de de hombre son violados en las cárceles y tristemente y lo dice las estadísticas",
                "el gen el adn del hombre no tiene que ver con que sea un violador un hombre no viola",
                "yo creo que es la cancion más pelotuda que escuche en toda mi vida se supone que tienen que dar un mensaje para todos",
                "el termino feminazi no existe y no no me enojo que estes en desacuerdo con el feminismo",
                "el problema viene cuando quieres denigrarlo y mofarte de ello",
                "los violadores con insomnio se ven esto para dormir agusto",
                "literalmente estas diciendo que las matan por ser mujeres te das cuenta de lo estúpido que suena eso",
                "ustedes mismas denigran su genero inventándose esas mierdas",
                "los machitos quieren aprovecharse de que son mujeres para violarlas",
                "no me jodas violación es violación no importa si es mujer o no",
                "un violador es una persona enferma y no le importara si su victima es mujer niño un adolescente etc",
                "esa chica se emborracho en una discoteca y pendejos aprovecharon para violarla"
            ]

    output_path = "artifacts/sample_predictions.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(sample_sentences):
            preds = annotate_examples(best_model, text, word_vocab, label_vocab, device)
            f.write(f"Example {i+1}\n")
            f.write(f"Text: {text}\n")
            f.write(f"Prediction: {preds}\n")
            f.write("-" * 80 + "\n")

    print(f"\nPredictions written to {output_path}")

    summary_path = os.path.join("artifacts", f"best_model_summary.txt")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Best Model Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {overall_best_model_name}\n")
        f.write(f"Best Hyperparameters:\n")
        for k, v in best_hparams.items():
            f.write(f"   - {k}: {v}\n")
        f.write("\n📊 Best Metrics:\n")
        for k, v in overall_best_metrics.items():
            if k != "confusion_matrix":
                f.write(f"   - {k}: {v:.4f}\n")

        f.write("\nConfusion Matrix:\n")
        matrix = overall_best_matrix
        for row in matrix:
            f.write("   " + "\t".join(map(str, row)) + "\n")

    print(f"\nSummary of best model written to {summary_path}")

if __name__ == "__main__":
    main()
