## Slurm Job Exporter Installation

1. Clone this repository in `/opt` or another location.
2. Create a python virtual environment, and install the packages in `requirements.txt`. nvidia-ml-py is only needed to monitor GPU usage.
3. Adjust the paths in `cpu-job-exporter.service` to run the python script.
4. The service can be enabled and started at this point.


## Prometheus Installation

1. Download [prometheus](https://prometheus.io/download/) and extract it to a desired location.
2. Configure the hostnames and ports for the exporters in `slurm-job-exporter/config/prometheus-config.yml`. More information on running prometheus with a config file can be found [here](https://prometheus.io/docs/introduction/first_steps/#configuring-prometheus).
3. Adjust the paths in `systemd/prometheus.service` to point to the prometheus executable and the configuration file.
4. The service can now be enabled and started. By default, the server will listen on localhost:9090, but can be changed in the service file.


## Grafana Installation
Grafana can be installed using yum, and automatically creates a service file in `/usr/lib/systemd/system/grafana-server.service`, which can be started using `systemctl enable --now grafana-server`. By default, the Grafana server will listen on localhost:3000.


## Grafana Configuration

The default location for the grafana configuration file is `/etc/grafana/grafana.ini`.

- [Grafana Docs: Configuring Grafana](https://grafana.com/docs/grafana/latest/setup-grafana/configure-grafana/)


## Prometheus Storage

Prometheus stores data in a database. The location of this data and the amount of time data is retained can be configured using the flags `--storage.tsdb.path=/path/to/data` and `--storage.tsdb.retention.time=15d`. You can also set the maximum number of bytes of storage blocks to retain using `--storage.tsdb.retention.size=5GB`. More information on storage can be found [here](https://prometheus.io/docs/prometheus/latest/storage/).