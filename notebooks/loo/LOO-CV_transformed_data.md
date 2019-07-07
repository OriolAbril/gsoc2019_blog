
Blog post exploring whether or not LOO-CV can be used to compare models that try to explain some data $y$ with models trying to explain the same data after a transformation $z=f(y)$. Inspired by [@tiagocc question](https://discourse.mc-stan.org/t/very-simple-loo-question/9258) on Stan Forums. This post has two sections, the first one is the mathematical derivation of the equations used and their application on a validation example, and the second section is a real example. In addition to the LOO-CV usage examples and explanations, another goal of this notebook is to show and highlight the capabilities of [ArviZ](https://arviz-devs.github.io/arviz/).

This post has been automatically generated from a Jupyter notebook that can be downloaded [here]({{ site.url }}/notebooks/loo/LOO-CV_transformed_data.ipynb)


```python
import pystan
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
```


```python
plt.style.use('../forty_blog.mplstyle')
```

## Mathematical derivation and validation example
In the first example, we will compare two equivalent models:

1. $y \sim \text{LogNormal}(\mu, \sigma)$
2. $\log y \sim \text{Normal}(\mu, \sigma)$

### Model definition and execution
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
with open("lognormal.stan", "r") as f:
    lognormal_code = f.read()
```

<details>
<summary markdown='span'>Stan code for LogNormal model
</summary>


```python
print(lognormal_code)
```

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



</details><br/>


```python
sm_lognormal = pystan.StanModel(model_code=lognormal_code)
fit_lognormal = sm_lognormal.sampling(data=data, iter=1000, chains=4)
```

    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_fa0385baccb7b330f85e0cacaa99fa9d NOW.



```python
idata_lognormal = az.from_pystan(
    posterior=fit_lognormal,
    posterior_predictive='y_hat',
    observed_data=['y'],
    log_likelihood='log_lik',
)
```


```python
with open("normal_on_log.stan", "r") as f:
    normal_on_log_code = f.read()
```

<details>
<summary markdown='span'>Stan code for Normal on Log data model
</summary>


```python
print(normal_on_log_code)
```

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



</details><br/>


```python
sm_normal = pystan.StanModel(model_code=normal_on_log_code)
fit_normal = sm_normal.sampling(data=data, iter=1000, chains=4)
```

    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_6b25918853568e528afbe629c1103e09 NOW.



```python
idata_normal = az.from_pystan(
    posterior=fit_normal,
    posterior_predictive='logy_hat',
    observed_data=['logy'],
    log_likelihood='log_lik',
)
```

Check model convergence. Use `az.summary` to in one view that the effective sample size (ESS) is large enough and $\hat{R}$ is close to one.


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
      <td>2.151</td>
      <td>0.218</td>
      <td>1.753</td>
      <td>2.549</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>1338.0</td>
      <td>1285.0</td>
      <td>1382.0</td>
      <td>1221.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>1.204</td>
      <td>0.168</td>
      <td>0.901</td>
      <td>1.510</td>
      <td>0.005</td>
      <td>0.004</td>
      <td>1090.0</td>
      <td>1066.0</td>
      <td>1126.0</td>
      <td>1002.0</td>
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
      <td>2.154</td>
      <td>0.222</td>
      <td>1.731</td>
      <td>2.568</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>1402.0</td>
      <td>1386.0</td>
      <td>1421.0</td>
      <td>1201.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sigma</th>
      <td>1.194</td>
      <td>0.160</td>
      <td>0.902</td>
      <td>1.492</td>
      <td>0.004</td>
      <td>0.003</td>
      <td>1333.0</td>
      <td>1273.0</td>
      <td>1428.0</td>
      <td>1067.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



In addition, we can plot the quantile ESS plot for one of them directly with `plot_ess`


```python
az.plot_ess(idata_normal, kind="quantile", color="k");
```


![png]({{ site.url }}/notebooks/loo/LOO-CV_transformed_data_files/LOO-CV_transformed_data_22_0.png)


### Posterior validation
Check that both models are equivalent and do indeed give the same result for both parameters.


```python
az.plot_posterior(idata_lognormal);
```


![png]({{ site.url }}/notebooks/loo/LOO-CV_transformed_data_files/LOO-CV_transformed_data_24_0.png)



```python
az.plot_posterior(idata_normal);
```


![png]({{ site.url }}/notebooks/loo/LOO-CV_transformed_data_files/LOO-CV_transformed_data_25_0.png)


### Calculate LOO-CV
Now we get to calculate LOO-CV using Pareto Smoothed Importance Sampling as detailed in Vehtari et al., 2017. As we explained above, both models are equivalent, but one is in terms of $y$ and the other in terms of $\log y$. Therefore, their likelihoods will be on different scales, and hence, their expected log predictive density will also be different.


```python
az.loo(idata_lognormal)
```




    Computed from 2000 by 30 log-likelihood matrix

           Estimate       SE
    IC_loo   226.00    14.38
    p_loo      2.05        -




```python
az.loo(idata_normal)
```




    Computed from 2000 by 30 log-likelihood matrix

           Estimate       SE
    IC_loo    96.66     8.71
    p_loo      2.00        -



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

In order to estimate the elpd loo for $y$ from the data in the second model, $z$, we have to describe $p_y(y)$ as a function of $z$ and $p_z(z)$. We know that $y$ and $z$ are actually related, and we can use this relation to find how would the random variable $y$ (which is actually a transformation of the random variable $z$) be distributed. This is done with the Jacobian. Therefore:

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
    IC_loo   225.84    14.46
    p_loo      2.00        -



## Real example

We will now use as data a subsample of a [real dataset](https://docs.google.com/spreadsheets/d/1gt1Dvi7AnQJiBb5vKaxanTis_sfd4sC4sVoswM1Fz7s/pub#). The subset has been selected using:

```python
df = pd.read_excel("indicator breast female incidence.xlsx").set_index("Breast Female Incidence").dropna(thresh=20).T
df.to_csv("indicator_breast_female_incidence.csv")
```

Below, the data is loaded and plotted for inspection.


```python
df = pd.read_csv("indicator_breast_female_incidence.csv", index_col=0)
df.plot();
```


![png]({{ site.url }}/notebooks/loo/LOO-CV_transformed_data_files/LOO-CV_transformed_data_37_0.png)


In order to show different examples of LOO on transformed data, we will take into account the following models:

$$
\begin{align}
&y=a_1 x+a_0 \\
&y=e^{b_0}e^{b_1 x}  &\rightarrow& \quad\log y = z_1 = b_1 x + b_0\\
&y=c_1^2 x^2 + 2 c_1 c_2 x + c_0^2  &\rightarrow& \quad\sqrt{y} = z_2 = c_1 x + c_0
\end{align}
$$

This models have been chosen mainly because of their simplicity. In addition, they can all be applied using the same Stan code and the data looks kind of linear. This will put the focus of the example on the loo calculation instead of on the model itself. For the online example, the data from Finland has been chosen, but feel free to download the notebook and experiment with it.


```python
y_data = df.Finland
z1_data = np.log(y_data)
z2_data = np.sqrt(y_data)
x_data = df.index/100 # rescale to set both to a similar scale
dict_y = {"N": len(x_data), "y": y_data, "x": x_data}
dict_z1 = {"N": len(x_data), "y": z1_data, "x": x_data}
dict_z2 = {"N": len(x_data), "y": z2_data, "x": x_data}
coords = {"year": x_data}
dims = {"y": ["year"], "log_likelihood": ["year"]}
```


```python
with open("linear_regression.stan", "r") as f:
    lr_code = f.read()
```

<details>
<summary markdown='span'>Stan code for Linear Regression
</summary>


```python
print(lr_code)
```

    data {
      int<lower=0> N;
      vector[N] x;
      vector[N] y;
    }

    parameters {
      real b0;
      real b1;
      real<lower=0> sigma_e;
    }

    model {
      b0 ~ normal(0, 20);
      b1 ~ normal(0, 20);
      for (i in 1:N) {
        y[i] ~ normal(b0 + b1 * x[i], sigma_e);
      }

    }

    generated quantities {
        vector[N] log_lik;
        vector[N] y_hat;
        for (i in 1:N) {
            log_lik[i] = normal_lpdf(y[i] | b0 + b1 * x[i], sigma_e);
            y_hat[i] = normal_rng(b0 + b1 * x[i], sigma_e);
        }
    }



</details><br/>


```python
sm_lr = pystan.StanModel(model_code=lr_code)
control = {"max_treedepth": 15}
```

    INFO:pystan:COMPILING THE C++ CODE FOR MODEL anon_model_48dbd7ee0ddc95eb18559d7bcb63f497 NOW.



```python
fit_y = sm_lr.sampling(data=dict_y, iter=1500, chains=6, control=control)
```


```python
fit_z1 = sm_lr.sampling(data=dict_z1, iter=1500, chains=6, control=control)
```


```python
fit_z2 = sm_lr.sampling(data=dict_z2, iter=1500, chains=6, control=control)
```

<details>
<summary markdown='span'>Convertion to InferenceData and posterior exploration
</summary>


```python
idata_y = az.from_pystan(
    posterior=fit_y,
    posterior_predictive='y_hat',
    observed_data=['y'],
    log_likelihood='log_lik',
    coords=coords,
    dims=dims,
)
idata_y.posterior = idata_y.posterior.rename({"b0": "a0", "b1": "a1"})
az.plot_posterior(idata_y);
```


![png]({{ site.url }}/notebooks/loo/LOO-CV_transformed_data_files/LOO-CV_transformed_data_49_0.png)



```python
idata_z1 = az.from_pystan(
    posterior=fit_z1,
    posterior_predictive='y_hat',
    observed_data=['y'],
    log_likelihood='log_lik',
    coords=coords,
    dims=dims,
)
az.plot_posterior(idata_z1);
```


![png]({{ site.url }}/notebooks/loo/LOO-CV_transformed_data_files/LOO-CV_transformed_data_50_0.png)



```python
idata_z2 = az.from_pystan(
    posterior=fit_z2,
    posterior_predictive='y_hat',
    observed_data=['y'],
    log_likelihood='log_lik',
    coords=coords,
    dims=dims,
)
idata_z2.posterior = idata_z2.posterior.rename({"b0": "c0", "b1": "c1"})
az.plot_posterior(idata_z2);
```


![png]({{ site.url }}/notebooks/loo/LOO-CV_transformed_data_files/LOO-CV_transformed_data_51_0.png)


</details><br/>

In order to compare the out of sample predictive accuracy, we have to apply the Jacobian transformation to the 2 latter models, so that all of them are in terms of $y$.

Note: we will use LOO instead of Leave Future Out algorithm even though it may be more appropriate because the Jacobian transformation to be applied is the same in both cases. Moreover, PSIS-LOO does not require refitting, and it is already implemented in ArviZ.

The transformation to apply to the second model $z_1 = \log y$ is the same as the previous example:


```python
old_loo_z1 = az.loo(idata_z1).loo
old_like = idata_z1.sample_stats.log_likelihood
idata_z1.sample_stats["log_likelihood"] = -z1_data.values+old_like
```

In the case of the third model, $z_2 = \sqrt{y}$:

$$ |\frac{dz}{dy}| = |\frac{1}{2\sqrt{y}}| = \frac{1}{2 z_2} \quad \rightarrow \quad \log |\frac{dz}{dy}| = -\log (2 z_2)$$


```python
old_loo_z2 = az.loo(idata_z2).loo
old_like = idata_z2.sample_stats.log_likelihood
idata_z2.sample_stats["log_likelihood"] = -np.log(2*z2_data.values)+old_like
```


```python
az.loo(idata_y)
```




    Computed from 4500 by 46 log-likelihood matrix

           Estimate       SE
    IC_loo   388.43     7.72
    p_loo      1.56        -




```python
print("LOO before Jacobian transformation: {:.2f}".format(old_loo_z1))
print(az.loo(idata_z1))
```

    LOO before Jacobian transformation: -141.43
    Computed from 4500 by 46 log-likelihood matrix

           Estimate       SE
    IC_loo   200.56     8.95
    p_loo      3.03        -



```python
print("LOO before Jacobian transformation: {:.2f}".format(old_loo_z2))
print(az.loo(idata_z2))
```

    LOO before Jacobian transformation: -4.93
    Computed from 4500 by 46 log-likelihood matrix

           Estimate       SE
    IC_loo   229.84     7.56
    p_loo      2.83        -


## References
Vehtari, A., Gelman, A., and Gabry, J. (2017):  Practical Bayesian Model Evaluation Using Leave-One-OutCross-Validation and WAIC, _Statistics and Computing_, vol. 27(5), pp. 1413–1432.
