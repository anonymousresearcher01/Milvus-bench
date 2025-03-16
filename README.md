# fluffy-moti
The experimental indications for motivating us to conduct research

# How this repo works

This is built around an motivational materials that stores a set of scripts, and provides a data (results).
The feature of the repo has been shared with some teams, such as distributing VectorDB or offloading VectorDB to DPU or SSD. To avoid unnecessary code conflict in the working scenarios, it is desirable to create the own folder and keep working on that.
The cleaning the invalid files up or arranging the whole material will be done later.

## Run pre-commit after code modification

To install `pre-commit`, use

```bash
pip install pre-commit
pre-commit install
```

To format and lint the python-based code, use
```bash
pre-commit
```

## Use Commitizen when you commit the code

To install `commitizen`, use

```bash
pip install --user -U commitizen
```

To commit, use
```bash
cz c
```

Refer to [Commitizen](https://commitizen-tools.github.io/commitizen/commands/commit/).

## Accessing the Dataset

The experimenal dataset may be too huge to share.
This could be possible to share the link for other member to download from official dataset website or Google drive, etc.
