# Scikit-Learn Extensions

## Description

sklearn-extensions is a library that implements some miscelaneous machine learning models and other functionality that wouldn't fit into the base sklearn repository.

## Installation

To install from source:

```bash
git clone https://github.com/eugenioLR/sklearn-extensions.git
cd sklearn-extensions
pip install -e .
```

## Components

The package is structured into the following modules:

| Module          | Description                                                                  |
|-----------------|------------------------------------------------------------------------------|
| `preprocessing` | Miscellaneous Feature transformers                                           |
| `models`        | Miscellaneous machine learning models.                                       |
| `wrappers`      | Meta-estimators for composing machine learning models and feature transformers.|
| `model_zoo`     | Module containing all the sklearn estimators in one file for easy accessing. |


## Usage

All classes in this package expose the standard scikit-learn interface (`fit`, `transform`, `predict`, etc.) and can be used interchangeably with native scikit-learn components in pipelines and grid searches.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request on [GitHub](https://github.com/eugenioLR/sklearn-extensions). Ensure that new components adhere to scikit-learn API conventions and include appropriate unit tests.

Since this repository is meant as a collection of miscellanous modules for machine learning, the novelty and applicability of the proposals will not be as relevant of a requirement for inclusion in the repository.

## License

This project is distributed under the terms of the GPL3 Public License. See the [LICENCE](LICENCE) file for details.
