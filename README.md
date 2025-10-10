Brenier Isotonic Regression
====

An OT-based solution to _cyclically monotone isotonic regression_,
an extension of isotonic regression to address multinomial inputs/outputs.
We consider the isotonic regression problem, where a regression function always has a cyclically monotone graph.

![](/images/cmir.png)

This expectedly performs nicely for multiclass calibration.
Below is the calibration map [1] produced by Brenier isotonic regression, where the base multiclass classifier is `sklearn.neural_network.MLPClassifier`.

![](/images/calib_map.png)

Some of the codes were ported from the Dirichlet calibration paper [2].
We sincerely appreciate [their well-organized code repository](https://github.com/dirichletcal/experiments_neurips).


## Dependencies

```sh
pip install numpy scipy scikit-learn matplotlib pot
```

## Try with a synthetic data

```sh
python main_synthetic.py \
  --bins [NUMBER_OF_BINS] \
  -n [SAMPLE SIZE] \
  -d [DIMENSION]
```

Let's try the univariate experiment first:
```sh
python main_synthetic.py -n 50 -d 1
```


## Try with OpenML data

```sh
python main.py \
  --bins [NUMBER_OF_BINS] \
  --data [balance-scale|car|cleveland|dermatology|glass|vehicle]
```
For the first time, it's reasonable to try this to see ECE metrics:
```sh
python main.py --bins 15 --data balance-scale
```
With `--vis` option, you can generate the calibration map:
```sh
python main.py --bins 15 --data balance-scale --vis
```


## References

- [1] Vaicenavicius et al. "Evaluating model calibration in classification" (AISTATS2019).
- [2] Kull et al. "Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration" (NeurIPS2019)