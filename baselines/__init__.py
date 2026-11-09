"""Published sleep-staging models run on ANPHY under their own and our configuration.

Each model is a sub-package with its upstream code, a summary of its paper and
one YAML file per configuration. Nothing here imports the GraphChangeNet
components or model; the data protocol, the split and the scorer are shared
through :mod:`baselines.common`.
"""
