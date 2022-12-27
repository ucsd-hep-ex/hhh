# hhh

## 1. Pull and start the Docker container
```bash
docker pull jmduarte/hhh
docker run -it jmduarte/hhh bash
```

## 2. Check out the GitHub repository
```bash
cd work
git clone https://github.com/ucsd-hep-ex/hhh
```

## 3. Install the Python package(s)
```bash
cd hhh
pip install -e .
cd ..
```

## 4. Copy and convert the dataset(s)
Copy the Delphes ROOT TTree datasets from:
- CERN EOS: `/eos/user/m/mstamenk/CxAOD31run/hhh-6b/delphes-samples/GF_HHH_SM_c3_0_d4_0_14TeV/sample_*.root`, or
- UCSD UAF: `/ceph/cms/store/user/woodson/GF_HHH_SM_c3_0_d4_0_14TeV/sample_*.root`

to the `data/delphes/v2/GF_HHH_SM_c3_0_d4_0_14TeV` directory

Convert to training and testing HDF5 files.
```bash
python -m src.data.delphes.convert_to_h5 data/delphes/v2/GF_HHH_SM_c3_0_d4_0_14TeV/sample_*.root --out-file data/delphes/v2/hhh_training.h5
python -m src.data.delphes.convert_to_h5 data/delphes/v2/GF_HHH_SM_c3_0_d4_0_14TeV/sample_*.root --out-file data/delphes/v2/hhh_testing.h5
```

## 5. Run the SPANet training
Override options file with `--gpus 0` if no GPUs are available.
```bash
python -m spanet.train -of options_files/delphes/hhh_v2.json [--gpus 0]
```

## 6. Evaluate the SPANet training
Assuming the output log directory is `spanet_output/version_0`.
Add `--gpu` if a GPU is available.
```bash
python -m spanet.test spanet_output/version_0 -tf data/delphes/v2/hhh_testing.h5 [--gpu]
```

## 7. Evaluate the baseline method
```bash
python -m src.models.test_baseline --test-file data/delphes/v2/hhh_testing.h5
```
