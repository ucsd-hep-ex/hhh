FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest
LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER $NB_USER
ENV USER=${NB_USER}

RUN sudo apt-get update \
    && sudo apt-get install -yq --no-install-recommends vim emacs \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

RUN pip install -q \
    coffea \
    xxhash \
    git+https://github.com/Alexanders101/SPANet@master \
    vector \
    mplhep \
    jetnet \
    pre-commit \
    jupyter_contrib_nbextensions \
    protobuf \
    pytorch-lightning \
    ray[tune] \
    hyperopt \
    jupyterlab_execute_time

# nbextensions not compatible with notebook>7 / jupyterlab
# RUN jupyter contrib nbextension install --user \
#     && jupyter nbextension enable execute_time/ExecuteTime

RUN fix-permissions /home/$NB_USER
