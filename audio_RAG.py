import csv
import os

import torch
from tqdm import tqdm

from config import MODELS_CED, N_BEST_RAG, N_CLASSES
from dataset import CachedEmbeddingDataset
from models import CedAudioClassifier

CLASS_NAMES = (
    "Alarm_bell_ringing",
    "Blender",
    "Cat",
    "Dishes",
    "Dog",
    "Electric_shaver_toothbrush",
    "Frying",
    "Running_water",
    "Speech",
    "Vacuum_cleaner",
)

def _rank_similar_vectors(embeddings, labels, query, top_k=10):
    """Return train vectors ranked by cosine similarity to a query."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    candidate_indices = torch.arange(len(embeddings), device=embeddings.device)
    similarities = torch.nn.functional.cosine_similarity(
        embeddings[candidate_indices], query.unsqueeze(0), dim=1
    )
    sorted_positions = torch.argsort(similarities, descending=True)[:top_k]

    results = []
    for position in sorted_positions:
        index = candidate_indices[position]
        results.append(
            {
                "index": index.item(),
                "vector": embeddings[index],
                "similarity": similarities[position].item(),
                "class": ";".join(
                    CLASS_NAMES[class_index]
                    for class_index in labels[index].nonzero(as_tuple=True)[0].tolist()
                ),
            }
        )
    return results


def top_similar_vectors(dataset, query_index=0, top_k=10):
    """Return the most similar cached vectors to one train query vector."""
    if not hasattr(dataset, "embeddings") or not hasattr(dataset, "labels"):
        raise ValueError("The dataset must contain cached embeddings and labels.")
    if not 0 <= query_index < len(dataset.embeddings):
        raise IndexError(f"query_index must be between 0 and {len(dataset.embeddings) - 1}.")

    candidates = torch.arange(len(dataset.embeddings)) != query_index
    return _rank_similar_vectors(
        dataset.embeddings[candidates],
        dataset.labels[candidates],
        dataset.embeddings[query_index],
        top_k,
    )


def top_similar_classes(embeddings, labels, query, number_of_classes):
    """Return classes ranked by the average of their best similarities."""
    if number_of_classes < 1:
        raise ValueError("number_of_classes must be at least 1.")
    if N_BEST_RAG < 1:
        raise ValueError("N_BEST_RAG must be at least 1.")
    if labels.ndim != 2 or labels.shape[1] != len(CLASS_NAMES):
        raise ValueError("labels must be multi-label tensors with one column per class.")

    similarities = torch.nn.functional.cosine_similarity(
        embeddings, query.unsqueeze(0), dim=1
    )
    class_results = []
    for class_index, class_name in enumerate(CLASS_NAMES):
        class_mask = labels[:, class_index].bool()
        if not class_mask.any():
            continue
        class_similarities = similarities[class_mask]
        best_similarities = torch.topk(
            class_similarities,
            k=min(N_BEST_RAG, class_similarities.numel()),
        ).values
        best_index = torch.where(class_mask)[0][class_similarities.argmax()]
        class_results.append(
            {
                "index": best_index.item(),
                "vector": embeddings[best_index],
                "similarity": best_similarities.mean().item(),
                "class": class_name,
            }
        )

    class_results.sort(key=lambda result: result["similarity"], reverse=True)
    return class_results[:number_of_classes]


def load_eval_labels(metadata_path):
    """Load and deduplicate the ground-truth labels for each eval file."""
    labels = {}
    with open(metadata_path, newline="") as metadata_file:
        for row in csv.DictReader(metadata_file, delimiter="\t"):
            filename = row["filename"]
            labels.setdefault(filename, [])
            if row["event_label"] not in labels[filename]:
                labels[filename].append(row["event_label"])
    return labels


def classify_eval_files(model, train_dataset, eval_dir, metadata_path, output_path):
    """Classify every eval file against the cached train embeddings."""
    eval_labels = load_eval_labels(metadata_path)
    train_embeddings = train_dataset.embeddings
    train_labels = train_dataset.labels
    model.eval()
    rows = []

    eval_files = sorted(
        filename for filename in os.listdir(eval_dir) if filename.lower().endswith(".wav")
    )
    with torch.no_grad():
        for filename in tqdm(eval_files, desc="Classifying eval"):
            inputs = model.preprocess_function(os.path.join(eval_dir, filename))
            query = model.encode(**inputs).squeeze(0).cpu()
            true_classes = eval_labels.get(filename)
            if not true_classes:
                raise ValueError(f"No ground-truth label found for eval file {filename}.")
            matches = top_similar_classes(
                train_embeddings,
                train_labels,
                query,
                number_of_classes=len(true_classes),
            )
            rows.append(
                {
                    "filename": filename,
                    "true_class": ";".join(true_classes),
                    "predicted_class": ";".join(match["class"] for match in matches),
                    "similarity": ";".join(
                        f"{match['similarity']:.6f}" for match in matches
                    ),
                }
            )

    with open(output_path, "w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["filename", "true_class", "predicted_class", "similarity"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    model_name = MODELS_CED[0]
    model = CedAudioClassifier(n_classes=N_CLASSES, model_name=model_name)
    data_root = os.path.join("data", "DESED_public_eval")
    embeddings_path = "embeddings"
    embeddings_file = os.path.join(
        embeddings_path, f"{model_name.split('/')[-1]}_multilabel_embeddings.pt"
    )
    os.makedirs(embeddings_path, exist_ok=True)
    dataset = CachedEmbeddingDataset(
        batch_size=8,
        num_frames=1,
        preprocess_function=model.preprocess_function,
        backbone=model,
        cache_path=embeddings_file,
    )
    classify_eval_files(
        model,
        dataset,
        os.path.join(data_root, "audio", "eval"),
        os.path.join(data_root, "metadata", "eval", "public_weak.tsv"),
        "audio_rag.csv",
    )


if __name__ == "__main__":
    main()
