# Bayesian Neural Network via Stochastic Gradient Descent
Code for the paper `Bayesian Neural Network via Stochastic Gradient Descent`.


Link to [Paper](https://abhinavsagar.github.io/files/bnn_sgd.pdf).

## Data

The dataset can be downloaded from [here](https://archive.ics.uci.edu/ml/datasets.php?format=&task=reg&att=&area=&numAtt=&numIns=&type=&sort=nameUp&view=table).

## Algorithm

![roc-auc](images/img1.png)

## Installation

`python bnn.py`

## Results
 
Bayesian neural network regression: average test RMSE
 
![roc-auc](images/img2.png)

Bayesian neural network regression: average negative test Log Likelihood

![roc-auc](images/img3.png)

## Citing

If you find this code useful in your research, please consider citing the paper:

```
@article{sagar2020bayesian,
  title={Bayesian Neural Network via Stochastic Gradient Descent},
  author={Sagar, Abhinav},
  journal={arXiv preprint arXiv:2006.08453},
  year={2020}
}
```

## License

```
MIT License

Copyright (c) 2020 Abhinav Sagar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
