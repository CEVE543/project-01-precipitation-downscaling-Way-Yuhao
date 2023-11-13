# Downscaling project

This github repository is a fork from a private [repo](https://github.com/Way-Yuhao/NCSN)https://github.com/Way-Yuhao/NCSN. Please refer to the private repo for full history of git commits. Permission required. 

This diffusion model is trained and evluated on MRMS and ERA5 data, and then evaluated on CPC data. Please refer to Canvas and [Doss-Gollin Lab Climate Data Repository](https://github.com/dossgollin-lab/climate-data) for instructions on how to download the data.

To train diffusion model, run `python train_conditional.py msg=MSG`, where MSG is a string to summurize this run. All training progresses are logged to Weights and Biases (wandb), including training and validation outputs on MRMS data.

To generate samples for evaluation, simply run `python rainfall_sr_cfg.py`, which will generate a batch of samples conditioned on artifically downsampled MRMS data. Alternativly, you can run `python sample_cpc_tx.py' to generate a batch of samples conditionaed on CPC data in the TX region. 

