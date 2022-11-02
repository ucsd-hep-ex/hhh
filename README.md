# hhh

## 1. Pull and start the Docker container
```bash
docker pull jmduarte/hhh
docker run -it jmduarte/hhh bash
```

## 2. Check out the GitHub repository with submodules
```bash
cd work
git clone https://github.com/ucsd-hep-ex/hhh --recurse-submodules
```

## 3. Install the Python package(s)
```bash
cd hhh
pip install -e .
cd SPANet
pip install -e .
cd ..
```

## 4. Download and convert the dataset(s)
Download ROOT TTree dataset `GluGluToHHHTo6B_SM.root` (Ask for the location) to the `data` directory

```bash
wget <URL> -O data/GluGluToHHHTo6B_SM.root
```

Convert to training and testing HDF5 files.
```bash
python src/data/convert_to_h5.py data/GluGluToHHHTo6B_SM.root --out-file data/hhh_training.h5
python src/data/convert_to_h5.py data/GluGluToHHHTo6B_SM.root --out-file data/hhh_testing.h5
```

## 5. Run the SPANet training
Override options file with `--gpus 0` if no GPUs are available.
```bash
python SPANet/spanet/train.py -of options_files/hhh.json [--gpus 0]
```

## 6. Evaluate the SPANet training
Assuming the output log directory is `spanet_output/version_0`.
Add `--gpu` if a GPU is available.
```bash
python SPANet/spanet/test.py spanet_output/version_0 -tf data/hhh_testing.h5 [--gpu]
```

## 7. Evaluate the baseline method
```bash
python src/models/test_baseline.py --test-file data/hhh_testing.h5
```
