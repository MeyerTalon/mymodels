# mymodels
models of mine

## essential commands

run these commands from the repository root:

### conda environment

```bash
# first time: create env and install all packages from environment.yaml
conda env create -f environment.yaml
```

```bash
# if the env already exists: sync installed packages to environment.yaml
conda env update -f environment.yaml --prune
```

```bash
# activate conda environment :)
conda activate mymodels
```

### tests

```bash
# run all tests
pytest
```

```bash
# run only the wikipedia tests
pytest wikipedia/tests
```

### wikipedia model

```bash
# train the wikipedia model
python wikipedia/training.py wikipedia/configs/wikipedia_small.yaml
```

```bash
# generate text with trained weights
python wikipedia/inference.py --model_name wikipedia_small --prompt "the history of"
```
