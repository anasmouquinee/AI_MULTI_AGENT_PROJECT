"""
data/download_papers.py
=======================
Script to download arXiv papers on few-shot learning in NLP.
Downloads 15 open-access PDFs to populate the corpus.
"""
import os
import time
import urllib.request
from pathlib import Path

# Destination directory
PAPERS_DIR = Path(__file__).resolve().parent / "papers"

# Curated list of arXiv papers on few-shot learning in NLP
# Format: (arxiv_id, descriptive_name)
PAPERS = [
    # --- Classic Few-shot learning ---
    ("2005.14165", "gpt3_language_models_few_shot_learners"),
    ("2001.07676", "meta_learning_nlp_survey"),
    ("2109.01652", "making_pre_trained_models_better_few_shot"),
    ("2012.15723", "making_pre_trained_models_few_shot_learners"),

    # --- Prompt-based learning ---
    ("2107.13586", "pre_train_prompt_predict_survey"),
    ("2103.10385", "exploiting_cloze_questions_few_shot"),
    ("2101.06804", "prefix_tuning_generation"),
    ("2104.08691", "power_of_scale_parameter_efficient"),

    # --- In-context learning ---
    ("2202.12837", "rethinking_role_demonstrations_icl"),
    ("2301.00234", "survey_in_context_learning"),
    ("2111.02080", "metaicl_learning_to_learn_in_context"),

    # --- Specific models and techniques ---
    ("2005.11401", "retrieval_augmented_generation_rag"),
    ("2009.01325", "few_shot_text_generation_with_pattern"),
    ("2106.13353", "lora_low_rank_adaptation"),
    ("2305.11206", "qlora_efficient_finetuning"),
]


def download_paper(arxiv_id: str, filename: str, output_dir: Path) -> bool:
    """Downloads a PDF from arXiv."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    filepath = output_dir / f"{filename}.pdf"

    if filepath.exists():
        print(f"  [OK] Already exists: {filename}.pdf")
        return True

    try:
        print(f"  Downloading: {filename}.pdf ...", end=" ", flush=True)
        urllib.request.urlretrieve(url, filepath)
        print("OK")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    """Downloads all papers."""
    print("=" * 60)
    print("Downloading arXiv papers for the RAG corpus")
    print("=" * 60)

    # Create directory if it doesn't exist
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    total = len(PAPERS)

    for i, (arxiv_id, name) in enumerate(PAPERS, 1):
        print(f"\n[{i}/{total}] {name}")
        if download_paper(arxiv_id, name, PAPERS_DIR):
            success_count += 1

        # Pause between downloads to avoid overloading arXiv
        if i < total:
            time.sleep(2)

    print(f"\n{'=' * 60}")
    print(f"Completed: {success_count}/{total} papers downloaded")
    print(f"Directory: {PAPERS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
