# Matrix Completion in the Unit Hypercube via Structured Matrix Factorization

This is the implementation of the approaches described in the paper:
> Emanuele Bugliarello, Swayambhoo Jain, and Vineeth Rakesh. [Matrix Completion in the Unit Hypercube via Structured Matrix Factorization](https://doi.org/10.24963/ijcai.2019/282). In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence (IJCAI-19), July 2019.

We provide the code for reproducing our results, as well as pre-trained models for the Click-Through Rate (CTR) data.

This code is based on a fork of [npMF](https://github.com/e-bug/npmf). Our methods for the unit interval, *Expertise Matrix Factorization* (EMF) and *Survival Matrix Factorization* (SMF), can be found in [`npmf/unit_models.py`](npmf/unit_models.py).

## Requirements
The requirements can be installed in either of the following ways:

- By setting up a dedicated conda environment: <br>
`conda env create -f environment.yml` followed by `source activate unitmf`
- With pip: <br>
`pip install -r requirements.txt`

## Data Preparation
Our CTR data is extracted from the [Outbrain Click Prediction competition](https://www.kaggle.com/c/outbrain-click-prediction/data).

Pre-processing steps and scripts to obtain the same Cross-Validation splits used in our paper are available in [`experiments/CTR/prepare_data/`](experiments/CTR/prepare_data).

You can also download the final splits from Google Drive: https://drive.google.com/drive/folders/1I_NjBXwAk1aBSDdFpCh4bWxsiXTjZ5bi

## Training

Scripts for training each model are provided in [`experiments/CTR/training_scripts/`](experiments/CTR/training_scripts).
- Default data directory: `experiments/CTR/data/`
- Default checkpoints directory: `experiments/CTR/checkpoints/`

### Best Models
The hyper-parameters used to obtain the results reported in the paper are listed in [`experiments/CTR/training_scripts/README.md`](experiments/CTR/training_scripts/README.md).

You can also download the final checkpoints from Google Drive: https://drive.google.com/drive/folders/1mJWuj08ae8uBF07jb52zGu55rbT4fr-O

## Evaluation

You can use the `test()` method in [`utils/eval.py`](utils/eval.py) to evaluate a model.

An example, along with the results reported in our paper, can be found in [`experiments/CTR/notebooks/results.ipynb`](experiments/CTR/notebooks/results.ipynb).

## License
This work is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). See [`LICENSE`](LICENSE) for details. 
Third-party datasets are subject to their respective licenses. <br>
If you find our code/models or ideas useful in your research, please consider citing the paper:
```
@inproceedings{Bugliarello+:IJCAI2019,
  title     = {Matrix Completion in the Unit Hypercube via Structured Matrix Factorization},
  author    = {Bugliarello, Emanuele and Jain, Swayambhoo and Rakesh, Vineeth},
  booktitle = {Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, {IJCAI-19}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {2038--2044},
  year      = {2019},
  month     = {7},
  location  = {Macao, China},
  doi       = {10.24963/ijcai.2019/282},
  url       = {https://doi.org/10.24963/ijcai.2019/282},
}
```
