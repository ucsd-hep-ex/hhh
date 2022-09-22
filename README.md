# hhh

1. Pull and start the Docker container:
```bash
docker pull jmduarte/hhh
docker run -it jmduarte/hhh bash
```

2. Check out this GitHub repository:
```bash
cd work
git clone git@github.com:ucsd-hep-ex/hhh --recurse-submodules
```

3. Install the Python package(s):
```bash
cd hhh
pip install -e .
cd SPANet
pip install -e .
```

4. Download and convert the dataset(s):
Download ROOT TTree dataset `GluGluToHHHTo6B_SM.root` (Ask for location).

Convert to training and testing HDF5 files.
```bash
python src/data/convert_to_h5.py --out-file hhh_training.h5
python src/data/convert_to_h5.py --out-file hhh_testing.h5
```

5. Run the training:
```bash
python SPANet/train.py -of SPANet/options_files/hhh.json
```

6. Evaluate the training (if output log directory is `spanet_output/version_0`):
```bash
python test.py spanet_output/version_0 -tf data/hhh_testing.h5
```

7. Evaluate the baseline:
```bash
python src/models/test_baseline.py
```
