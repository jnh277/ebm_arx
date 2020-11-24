# Deep Energy-Based NARX Models

Companion code to the paper:
```
Deep Energy-Based NARX Models
```

The paper gives four examples of using deep Energy Based Models to predict the conditional
distribution of the outputs given that the relationship between the current output and
past outputs and inputs is modelled using a nonlinear autoregressive eXogenous (NARX) structure.

- A simulated scalar AR example with four different noise models
    - Gausisan noise
    - Bimodal Gaussian noise
    - Cauchy noise
    - Gaussian noise with variance dependent on y_{t-1}
- A simulated second order linear ARX model
- A simulated nonlinear ARX example using the model presented by BLAH
- A real data example using the CE8 coupled electric drives benchmark data set.

## Simulated scalar AR example
This example is described in Section 5.1 of the paper.

To use saved results skip this step. The results for this example can be generated by running 
```bash
python scalar_example.py --Model
```

TODO: Add model options to the script

The results can then be plotted by running
```bash
python plot_scalar_example.py --model
```
TODO: add model options to this script

| Gaussian | Bimodal |
| 

TODO: add images

## Simulated linear ARX example
This example is described in section 5.2 of the paper.

To use saved results skip this step. The results can be regenerated by running
```bash
python arx_example.py
```

The results can be plotted by running
```bash
python plot_arx_example.py
```

## Simulated nonlinear example
This example is described in section 5.3 of the paper.

To use saved results skip this step. The results can be regenerated by running
```bash
python chen_arx_example.py
```

The results can be plotted by running
```bash
python plot_chen_example.py
```

## Real data: CE8 coupled electric drives benchmark
This example is described in section 5.3 of the paper.

To use saved results skip this step. The results can be regenerated by running
```bash
python coupledElectricDrives.py
```

The results can be plotted by running
```bash
python plot_CED_example.py
```


