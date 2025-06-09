# hhh

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14248960.svg)](https://doi.org/10.5281/zenodo.14248960)

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

# Instructions for CMS data set baseline
The CMS dataset was updated to run with the `v26` setup (`nAK4 >= 4 and HLT selection`). The update includes the possibility to apply the b-jet energy correction. By keeping events with at a least 4 jets, the boosted training can be performed on a maximum number of events and topologies.

List of samples (currently setup validated using 2018):
```
/eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2016APV.root
/eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2016.root
/eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2017.root
/eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2018.root
```

To run the framework, first convert the samples (this will allow to use both jets `pt` or `ptcorr`, steerable from the configuration file:
```
mkdir data/cms/v26/
python -m src.data.cms.convert_to_h5 /eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2018.root --out-file data/cms/v26/hhh_training.h5
python -m src.data.cms.convert_to_h5 /eos/user/m/mstamenk/CxAOD31run/hhh-6b/cms-samples-spanet/v26/GluGluToHHHTo6B_SM_spanet_v26_2018.root --out-file data/cms/v26/hhh_testing.h5
```

Then training can be done via:

```
python -m spanet.train -of options_files/cms/hhh_v26.json --gpus 1
```

Two config files exist for the event options:
```
event_files/cms/hhh.yaml # regular jet pT
event_files/cms/hhh_bregcorr.yaml # jet pT with b-jet energy correction scale factors applied
```

Note: to run the training with the b-jet energy correction applied, the `log_normalize` of the input variable was removed. Keeping it caused a 'Assignement collision'.
