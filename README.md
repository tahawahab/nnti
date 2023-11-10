# Getting Started

## Installation of Miniconda

### Step 1: Install Miniconda
To initiate the setup, begin by installing Miniconda. Execute the following command:

```bash
condor_submit setup.sub
```

This command installs Miniconda in the directory `~/miniconda3` and includes all the packages listed in `environment.yml`. Should you require additional packages, you can easily incorporate them by adding them to the `environment.yml` file and re-executing the setup job.

### Step 2: Configuring the System Path
To integrate `conda` into your system path, append the following line to your `~/.bashrc` file:

```bash
export PATH=$HOME/miniconda3/bin:$PATH
```

Afterwards, activate the changes by sourcing the `~/.bashrc` file:

```bash
source ~/.bashrc
```

### Step 3: Submitting Scripts to the Cluster
With Miniconda configured, you can now submit scripts to the cluster. For instance, to submit the `torch_matmul_docker.py` script, use the following command:

```bash
condor_submit run.sub
```

This process triggers `conda_run.sh`, which in turn selects the Conda environment as specified in the `environment.yml` file, and subsequently executes the script defined in `run.sh`. To modify the Python script being executed, simply edit the last line in the `run.sh` file.

## Monitoring Job Execution and Logs

### Accessing Job Logs
Upon the successful execution of a job, its log files are stored within the `logs/` directory. To review the output of a specific job, utilize the following command, replacing `<job_id>` with the actual ID of the job:

```bash
less logs/run.<job_id>.0.out
```

### Real-time Monitoring of Ongoing Processes
For real-time monitoring of a process that is currently in execution, the following command can be used, again substituting `<job_id>` with the correct job ID:

```bash
tail -f logs/run.<job_id>.0.out
```
