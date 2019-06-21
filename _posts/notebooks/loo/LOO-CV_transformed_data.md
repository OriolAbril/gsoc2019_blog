
# LOO-CV on transformed data
Blog post exploring whether or not LOO-CV can be used to compare models that try to explain some data $y$ with models trying to explain the same data after a transformation $z=f(y)$. Inspired by [@tiagocc question](https://discourse.mc-stan.org/t/very-simple-loo-question/9258) on Stan Forums. Intended to be a follow up to my answer there.

In the first example, we will compare two equivalent models: 

1. $y \sim \text{LogNormal}(\mu, \sigma)$
2. $\log y \sim \text{Normal}(\mu, \sigma)$


```python
import pystan
import pandas as pd
import numpy as np
import arviz as az
az.style.use('arviz-darkgrid')
import matplotlib.pyplot as plt
```

Define the data and execute the two models


```python
mu = 2
sigma = 1
logy = np.random.normal(loc=mu, scale=sigma, size=30)
y = np.exp(logy) # y will then be distributed as lognormal
data = {
    'N': len(y),      
    'y': y,
    'logy': logy
}
```


```python
lognormal_code = """
data {
  int<lower=0> N;
  vector[N] y;
}

parameters {
  real mu;
  real<lower=0> sigma;
}

model {
    y ~ lognormal(mu, sigma);
}

generated quantities {
    vector[N] log_lik;
    vector[N] y_hat;

    for (i in 1:N) {
        log_lik[i] = lognormal_lpdf(y[i] | mu, sigma);
        y_hat[i] = lognormal_rng(mu, sigma);
    }
}
"""
```


```python
sm_lognormal = pystan.StanModel(model_code=lognormal_code)
fit_lognormal = sm_lognormal.sampling(data=data, iter=1000, chains=4)
```

    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_3596002ce22937246377a0479b804226 NOW.



```python
idata_lognormal = az.from_pystan(
    posterior=fit_lognormal,
    posterior_predictive='y_hat',
    observed_data=['y'],
    log_likelihood='log_lik',
)
```


```python
normal_on_log_code = """
data {
  int<lower=0> N;
  vector[N] logy;
}

parameters {
  real mu;
  real<lower=0> sigma;
}

model {
    logy ~ normal(mu, sigma);
}

generated quantities {
    vector[N] log_lik;
    vector[N] logy_hat;

    for (i in 1:N) {
        log_lik[i] = normal_lpdf(logy[i] | mu, sigma);
        logy_hat[i] = normal_rng(mu, sigma);
    }
}
"""
```


```python
sm_normal = pystan.StanModel(model_code=normal_on_log_code)
fit_normal = sm_normal.sampling(data=data, iter=1000, chains=4)
```

    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_acd7c874588f1c862727f931f4dbf916 NOW.



```python
idata_normal = az.from_pystan(
    posterior=fit_normal,
    posterior_predictive='logy_hat',
    observed_data=['logy'],
    log_likelihood='log_lik',
)
```

Check model convergence


```python
az.summary(idata_lognormal)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hpd_3%</th>
      <th>hpd_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu</th>
      <td>2.038</td>
      <td>0.144</td>
      <td>1.774</td>
      <td>2.309</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>1290.0</td>
      <td>1290.0</td>
      <td>1373.0</td>
      <td>920.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>0.779</td>
      <td>0.109</td>
      <td>0.588</td>
      <td>0.981</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>1371.0</td>
      <td>1284.0</td>
      <td>1503.0</td>
      <td>1041.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
az.summary(idata_normal)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>hpd_3%</th>
      <th>hpd_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mu</th>
      <td>2.039</td>
      <td>0.141</td>
      <td>1.797</td>
      <td>2.314</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>1289.0</td>
      <td>1275.0</td>
      <td>1303.0</td>
      <td>979.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>0.772</td>
      <td>0.107</td>
      <td>0.585</td>
      <td>0.963</td>
      <td>0.003</td>
      <td>0.002</td>
      <td>1578.0</td>
      <td>1476.0</td>
      <td>1660.0</td>
      <td>1243.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Check that both models are equivalent and do indeed give the same result


```python
az.plot_posterior(idata_lognormal);
```


![png](LOO-CV_transformed_data_files/LOO-CV_transformed_data_14_0.png)



```python
az.plot_posterior(idata_normal);
```


![png](LOO-CV_transformed_data_files/LOO-CV_transformed_data_15_0.png)



```python
print(plt.rcParams["savefig.facecolor"])
print(plt.rcParams["savefig.transparent"])
```

    white
    False


# Calculate LOO-CV
Now we get to calculate LOO-CV using Pareto Smoothed Importance Sampling as detailed in Vehtari et al., 2017.


```python
az.loo(idata_lognormal, pointwise=True)
```




    Computed from 2000 by 30 log-likelihood matrix
    
           Estimate       SE
    IC_loo   192.66    11.15
    p_loo      1.76        -
    ------
    
    Pareto k diagnostic values:
                             Count   Pct.
    (-Inf, 0.5]   (good)       30  100.0%
     (0.5, 0.7]   (ok)          0    0.0%
       (0.7, 1]   (bad)         0    0.0%
       (1, Inf)   (very bad)    0    0.0%




```python
az.loo(idata_normal, pointwise=True)
```




    Computed from 2000 by 30 log-likelihood matrix
    
           Estimate       SE
    IC_loo    70.18     7.17
    p_loo      1.74        -
    ------
    
    Pareto k diagnostic values:
                             Count   Pct.
    (-Inf, 0.5]   (good)       30  100.0%
     (0.5, 0.7]   (ok)          0    0.0%
       (0.7, 1]   (bad)         0    0.0%
       (1, Inf)   (very bad)    0    0.0%



We have found that as expected, the two models yield different results despite being actually the same model. This is because. LOO is estimated from the log likelihood, $\log p(y_i\mid\theta^s)$, being $i$ the observation id, and $s$ the MCMC sample id. Following Vehtari et al., 2017, this log likelihood is used to calculate the PSIS weights and to estimate the expected log pointwise predictive density in the following way:

1. Calculate raw importance weights: $r_i^s = \frac{1}{p(y_i\mid\theta^s)}$
2. Smooth the $r_i^s$ (see original paper for details) to get the PSIS weights $w_i^s$
3. Calculate elpd LOO as:

$$ \text{elpd}_{psis-loo} = \sum_{i=1}^n \log \left( \frac{\sum_s w_i^s p(y_i|\theta^s)}{\sum_s w_i^s} \right) $$

This will estimate the out of sample predictive fit of $y$ (where $y$ is the data of the model. Therefore, for the first model, using a LogNormal distribution, we are indeed calculating the desired quantity:

$$ \text{elpd}_{psis-loo}^{(1)} \approx \sum_{i=1}^n \log p(y_i|y_{-i}) $$

Whereas for the second model, we are calculating:

$$ \text{elpd}_{psis-loo}^{(2)} \approx \sum_{i=1}^n \log p(z_i|z_{-i}) $$

being $z_i = \log y_i$. We actually have two different probability density functions, one over $y$ which from here on we will note $p_y(y)$, and $p_z(z)$.

In order to estimate the elpd loo for $y$ from the data in the second model, $z$, we have to describe $p_y(y)$ as a function of $z$ and $p_z(z)$. We know that $y$ and $z$ are actually related, and we can use this relation to find how would the random variable $y$ which is actually a transformation of the random variable $z$ be distributed. Therefore:

$$
p_y(y|\theta)=p_z(z|\theta)|\frac{dz}{dy}|=\frac{1}{|y|}p_z(z|\theta)=e^{-z}p_z(z|\theta)
$$

In the log scale:

$$
\log p_y(y|\theta)=-z + \log p_z(z|\theta)
$$

We apply the results to the log likelihood data of the second model (the normal on the logarithm instead of the lognormal) and check that now the result does coincide with the LOO-CV estimated by the lognormal model.


```python
old_like = idata_normal.sample_stats.log_likelihood
z = logy
idata_normal.sample_stats["log_likelihood"] = -z+old_like
```


```python
az.loo(idata_normal)
```




    Computed from 2000 by 30 log-likelihood matrix
    
           Estimate       SE
    IC_loo   192.53    11.23
    p_loo      1.74        -



## References
Vehtari, A., Gelman, A., and Gabry, J. (2017):  Practical Bayesian Model Evaluation Using Leave-One-OutCross-Validation and WAIC, _Statistics and Computing_, vol. 27(5), pp. 1413–1432.


```python

```
