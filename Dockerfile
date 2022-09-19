FROM gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp:latest
LABEL maintainer="Javier Duarte <jduarte@ucsd.edu>"

USER $NB_USER
ENV USER=${NB_USER}

RUN sudo apt-get update \
    && sudo apt-get install -yq --no-install-recommends vim emacs \
    && sudo apt-get clean && sudo rm -rf /var/lib/apt/lists/*

RUN mamba install -c pytorch -c conda-forge \
    pytorch=1.12.0 \
    torchvision \
    torchaudio \
    cudatoolkit=11.6

RUN pip install -q \
    tables \
    mplhep \
    jetnet \
    weaver-core \
    pre-commit \
    ogb \
    pytorch-lightning \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    pyg-nightly \
    jupyter_contrib_nbextensions \
    -f https://data.pyg.org/whl/torch-1.12.0+cu116.html

RUN jupyter contrib nbextension install --user \
    && jupyter nbextension enable execute_time/ExecuteTime

RUN fix-permissions /home/$NB_USER
