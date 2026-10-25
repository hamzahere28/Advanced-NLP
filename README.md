# Advanced NLP

**Muhammad Hamza** · [hamzahere28](https://github.com/hamzahere28)

TensorFlow implementations covering attention, positional encoding, encoder and decoder layers, Transformer training, and Portuguese-to-English inference.

## Modules

| Module | Topic | Python code | Notebook with outputs |
|---|---|---|---|
| 1 | Self attention and sentiment classification | [module_1.py](Module%201/module_1.py) | [module_1.ipynb](Module%201/module_1.ipynb) |
| 2 | Positional encoding | [module_2.py](Module%202/module_2.py) | [module_2.ipynb](Module%202/module_2.ipynb) |
| 3 | Encoder and decoder layers | [module_3.py](Module%203/module_3.py) | [module_3.ipynb](Module%203/module_3.ipynb) |
| 4 | Complete Transformer training | [module_4.py](Module%204/module_4.py) | [module_4.ipynb](Module%204/module_4.ipynb) |
| 5 | Autoregressive translation | [module_5.py](Module%205/module_5.py) | [module_5.ipynb](Module%205/module_5.ipynb) |

Each folder contains the complete sequential code, the notebook with recorded outputs, a text output log, and result figures. The Python scripts retain all code cells, including commented cells and repeated definitions.

## Running

```bash
git clone https://github.com/hamzahere28/Advanced-NLP.git
cd Advanced-NLP
python -m pip install -r requirements.txt
jupyter notebook
```

Run each notebook from top to bottom in module order. The scripts can also be executed with commands such as `python "Module 1/module_1.py"` after configuring resource paths.

The code uses Kaggle input paths. For a local run, point the sentiment loader to `Module 1/data`, download the TED language files and update their paths, and configure the Wiki-words embedding model path for Module 1. TensorFlow 2.19.0 was recorded in the saved runs. Models were not retrained during repository preparation.

## Dataset sources

- [Cornell Sentence Polarity v1.0](https://www.cs.cornell.edu/people/pabo/movie-review-data/): `rt-polarity.pos` and `rt-polarity.neg`, included under `Module 1/data`. The two files are parts of one dataset with 10,662 snippets.
- [TED Talks Transcripts for NLP](https://www.kaggle.com/datasets/miguelcorraljr/ted-ultimate-dataset): `ted_talks_pt-br.csv` and `ted_talks_en.csv`, joined by `talk_id`. These larger files must be attached or downloaded separately.
- [Google Wiki-words](https://www.kaggle.com/models/google/wiki-words) provides the pretrained embeddings used in Module 1.

## Recorded results

| Run | Epochs | Final training loss |
|---|---:|---:|
| Module 1 classifier | 5 | 0.6881 |
| Module 2 classifier | 5 | 0.6917 |
| Module 3 classifier | 5 | 0.6791 |
| Module 4 Transformer | 5 | 5.7127 |
| Module 5 tiny Transformer | 30 | 5.4352 |
| Module 5 larger Transformer | 30 | 2.3346 |

Training values for Modules 2 and 3 use the explicit history printouts, which differ from their saved progress bars. Both Module 5 models use the 500 loaded transcript pairs; some original chart labels are stale. Translation outputs are repetitive, and no held-out translation score is recorded. Attention masks are implemented, but the translation training loss does not explicitly mask padding positions.

![Translation output](Module%205/results/finalResult.PNG)
