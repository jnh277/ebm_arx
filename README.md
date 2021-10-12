# Deep Energy-Based NARX Models

Companion code to the paper:
```
Deep Energy-Based NARX Models https://arxiv.org/abs/2012.04136
```

The paper gives four examples of using deep Energy Based Models to predict the conditional
distribution of the outputs given that the relationship between the current output and
past outputs and inputs is modelled using a nonlinear autoregressive eXogenous (NARX) structure.

- A simulated scalar AR example with four different noise models (section 4.1 of paper)
    - Gausisan noise
        ```bash
        python scalar_example.py -m gaussian  # skip this line to use pregenerated results
        python plot_scalar_example.py -m gaussian
        ```
    - Bimodal Gaussian noise 
        ```bash
        python scalar_example.py -m bimodal  # skip this line to use pregenerated results
        python plot_scalar_example.py -m bimodal
        ```
    - Cauchy noise
        ```bash
        python scalar_example.py -m cauchy  # skip this line to use pregenerated results
        python plot_scalar_example.py -m cauchy
        ```
    - Gaussian noise with variance dependent on y_{t-1}
        ```bash
        python scalar_varying.py
        ```
- A simulated second order linear ARX model with Gaussian mixture noise (section 4.2 of paper)
    ```bash
    python arx_example.py # skip this line to use pregenerated results
    python plot_arx_example.py
    ```
- A simulated nonlinear ARX example (section 4.3 of paper)
    ```bash
    python chen_arx_example.py # skip this line to use pregenerated results
    python plot_chen_example.py
    ```
    The comparison of model performance for the hyperparameters described in the appendix
    can be run using: (warning this will take a long time, as such pregenerated results for 
    the combinations specified in the paper can be used by skipping this step)
    ```bash
    python chen_model_comparison.py -n [N] -s [Sigma]
    ```
    where [N] is the number of data points and [Sigma] is the noise level from 0.1 to 1.0
    The results can be evaluated by running:
    ```bash
    python load_chen_comparison_results.py -n [N] -s [Sigma]
    ```
    Pregenerated results are available for combinations of N={100,250,500} and Sigma = {0.1,0.3,0.5}
    
  
- A real data example using the CE8 coupled electric drives benchmark data set (section 4.4 of paper)
    ```bash
    python coupledElectricDrives.py
    python plot_CED_example.py
    ```
      The comparison of model performance for the hyperparameters described in the appendix
    can be run using: (warning this will take a long time, as such pregenerated results are available)
    ```bash
    python CED_comp.py 
    ```
    The results can be evaluated by running:
    ```bash
    python load_CED_comp.py
    ```

  




