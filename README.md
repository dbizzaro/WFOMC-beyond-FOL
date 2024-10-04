[![Paper](http://img.shields.io/badge/paper-arxiv.2308.11738-B31B1B.svg)](https://arxiv.org/abs/2308.11738)
# Lifted Inference beyond First-Order Logic
Code for Weighted First-Order Model Counting with DAG, Connected, Tree and Forest Axioms.

## Setup
Clone the repository and install the requirements:

```
git clone https://github.com/dbizzaro/WFOMC-beyond-FOL.git
cd WFOMC-beyond-FOL
conda create --name wfomc_env --file environment.txt
```

## Usage
In order to compute the WFOMC of something, we first need to generate an object of class `Language` and one of class `Formula`. Then, we can call the function:
```
WFOMC(language, formula, n)
```

The constructor of `Language` takes as inputs the list of unary predicates, the list of binary predicates and possibly a dictionary of weights (with default weights being equal to 1), as in:
```
Language(['P'], ['R'], {'P': (2,-1.5)})
```
The constructor of `Formula` takes as input the logical expression and possibly an axiom, the axiom predicate, a dictionary of cardinality constraints on unary predicates and one on binary predicates, as in:
```
Formula('(Px & ~Py) >> (Rxy | Ryx)', 'DAG', 'R', {'Px': 3}, {'R': ('>', 5)})
```
The available axioms are: `DAG`, `connected`, `tree`, `forest` and `none`.


## Bibtex
```
@misc{malhotra2024liftedinferencefirstorderlogic,
      title={Lifted Inference beyond First-Order Logic}, 
      author={Sagar Malhotra and Davide Bizzaro and Luciano Serafini},
      year={2024},
      eprint={2308.11738},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2308.11738}, 
}
```
