# Training scripts
Scripts for training each model.
- Default data directory: `../data/`
- Default checkpoints directory: `../checkpoints/`

We trained each model using the following hyper-parameters (whenever applicable):

-  `max_epochs = 100`
- `batch_size = 128`
- `decay_rate = 0.95`

## Best hyper-parameters

### K = 10

- MF (sgd_bias)
  >   lr = 1.0 <br>
lambda_u = 1e-6 <br>
lambda_i = 1e-4

- NMF (anls)
  > lr = 1.0 <br>
lambda_u = 1e-6 <br>
lambda_i = 1e-4 <br>
int_iter = 80

- PMF
  > lr = 1.0 <br>
lambda_u = 1e-6 <br>
lambda_i = 1e-4

- LMF
  > lr = 1.0 <br>
lambda_u = 1e-6 <br>
lambda_i = 1e-4

- EMF 
  > lr = 0.01 <br>
lambda_u = 4.641589e-06 <br>
lambda_i = 1.0 <br>
int_iter = 50

- SMF 
  > lr = 21.54435 <br>
lambda_u = 1.29155e-07 <br>
lambda_i = 1e-06

### K = 15

- MF (sgd_bias)
  >   lr = 1.0 <br>
lambda_u = 1e-6 <br>
lambda_i = 1e-4

- NMF (anls)
  > lr = 1.0 <br>
lambda_u = 1e-6 <br>
lambda_i = 1e-4 <br>
int_iter = 80

- PMF
  > lr = 1.0 <br>
lambda_u = 1e-6 <br>
lambda_i = 1e-4

- LMF
  > lr = 1.0 <br>
lambda_u = 1e-6 <br>
lambda_i = 1e-4

- EMF 
  > lr = 0.01 <br>
lambda_u = 4.641589e-06 <br>
lambda_i = 1.0 <br>
int_iter = 50

- SMF 
  > lr = 21.54435 <br>
lambda_u = 1.29155e-07 <br>
lambda_i = 1e-06

### K = 20

- MF (sgd_bias)
  >   lr = 1.0 <br>
lambda_u = 1e-6 <br>
lambda_i = 1e-4

- NMF (anls)
  > lr = 1.0 <br>
lambda_u = 1e-6 <br>
lambda_i = 1e-4 <br>
int_iter = 80

- PMF
  > lr = 1.0 <br>
lambda_u = 1e-6 <br>
lambda_i = 1e-4

- LMF
  > lr = 1.0 <br>
lambda_u = 1e-6 <br>
lambda_i = 1e-4

- EMF 
  > lr = 0.01 <br>
lambda_u = 4.641589e-06 <br>
lambda_i = 1.0 <br>
int_iter = 50

- SMF 
  > lr = 16.68101 <br>
lambda_u = 1e-10 <br>
lambda_i = 1.29155e-09
