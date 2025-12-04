## Slurm Job Exporter Installation

1. Clone this repository in `/opt` or another location.
2. Create a python virtual environment, and install the packages in `requirements.txt`. nvidia-ml-py is only needed to monitor GPU usage.
3. Adjust the paths in `cpu-job-exporter.service` to run the python script.
4. The service can be enabled and started at this point.


## Prometheus Installation

1. Download [prometheus](https://prometheus.io/download/) and extract it to a desired location.
2. Configure the hostnames and ports for the exporters in `slurm-job-exporter/config/prometheus-config.yml`. More information on running prometheus with a config file can be found [here](https://prometheus.io/docs/introduction/first_steps/#configuring-prometheus).
3. Adjust the paths in `systemd/prometheus.service` to point to the prometheus executable and the configuration file.
4. The service can now be enabled and started.