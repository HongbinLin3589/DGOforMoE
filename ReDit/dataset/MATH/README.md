---
license: mit
task_categories:
  - text-generation
language:
  - en
tags:
  - explanation-generation
pretty_name: Mathematics Aptitude Test of Heuristics (MATH) Dataset
size_categories:
  - 10K<n<100K
arxiv_id: 2103.03874
dataset_info:
  config_name: default
  splits:
    - name: train
      num_examples: 7500
    - name: test
      num_examples: 5000
---

# MATH Dataset

The Mathematics Aptitude Test of Heuristics (MATH) dataset consists of problems from mathematics competitions, including the AMC 10, AMC 12, AIME, and more. Each problem in MATH has a full step-by-step solution, which can be used to teach models to generate answer derivations and explanations.

This is a converted version of the [hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math) originally created by Hendrycks et al. The dataset has been converted to parquet format for easier loading and usage.

## Data Fields

- `problem`: The mathematics problem text
- `level`: Difficulty level of the problem (e.g., AMC 10, AMC 12, AIME)
- `type`: Type of mathematics problem (e.g., Algebra, Geometry, Counting & Probability)
- `solution`: Step-by-step solution to the problem

## Data Splits

The dataset contains two splits:
- `train`: Training set, 7500 problems
- `test`: Test set, 5000 problems

## Original Dataset Information

- **Original Dataset**: [hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math)
- **Paper**: [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874)
- **Homepage**: [https://github.com/hendrycks/math](https://github.com/hendrycks/math)

## Citation

If you use this dataset, please cite the original work:

```bibtex
@article{hendrycksmath2021,
    title={Measuring Mathematical Problem Solving With the MATH Dataset},
    author={Dan Hendrycks
    and Collin Burns
    and Saurav Kadavath
    and Akul Arora
    and Steven Basart
    and Eric Tang
    and Dawn Song
    and Jacob Steinhardt},
    journal={arXiv preprint arXiv:2103.03874},
    year={2021}
}
```

## License

This dataset follows the same license as the original dataset: [License](https://github.com/hendrycks/math/blob/main/LICENSE)