# Hack-RAG

Just my exploration on RAG, nothing special.

## Project Proposal
* RAG-powered data science assistant.
* EDA with LLMs.

## Data Preparation
1. Create folders `./data/raw/` and `./data/processed/`.
2. Download [Meta Kaggle](https://www.kaggle.com/datasets/kaggle/meta-kaggle/data) and put the zip file under `./data/raw/`.
3. Generate processed data by running:
    * [01_eda.ipynb](https://github.com/JiangJiaWei1103/Hack-RAG/blob/main/01_eda.ipynb) to get `forums.pqt` dumped under `./data/processed/`.
    * [02_preprocess.ipynb](https://github.com/JiangJiaWei1103/Hack-RAG/blob/main/02_preprocess.ipynb) to get `mini_forums.pqt` dumped under `./data/processed/`.

## Try a Quick RAG
Run the following command to see the RAG result in `demo.log`:
```
python3 -m first_rag
```


## References
* [Google - AI Assistants for Data Tasks with Gemma](https://www.kaggle.com/competitions/data-assistants-with-gemma/)
