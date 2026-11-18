# Job entry points

Slurm scripts for environment checks, weight download, fine-tuning, and generation.

Method: Jobs start with a cleared environment, an absolute interpreter, and the variables set inside the script. Formal training and generation each take one GPU. Scoring after a finished prediction does not call the model again.

| Script | Role |
| --- | --- |
| gpu_formal_train.sbatch | One BiomedBERT run, then its specified evaluation |
| gpu_formal_generate.sbatch | One frozen generation shard |
| cpu_job.sbatch | CPU checks and scoring |
| inspect_env.py | Confirms the job did not import the login environment |

Requesting 8 CPUs on this site still allocates the partition default. The scripts do not change the effective batch size.
