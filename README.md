INSTALLATION:

git clone https://github.com/Joxy97/HHH--6b.git

cd HHH--6b

pip3 install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1

pip3 install jupyterlab matplotlib scikit-hep

pip3 install -e .

git clone https://github.com/Alexanders101/SPANet.git

cd SPANet

pip3 install -e .

cd ..

pip3 install tensorboard rich
pip3 install pytorch-lightning==1.8.5


USAGE:

cp <your_data>.root ./data

python3 src/data/cms/convert_to_h5.py data/<your_data>.root --out-file data/hhh_training.h5
python3 src/data/cms/convert_to_h5.py data/<your_data>.root --out-file data/hhh_testing.h5

#modify ./event_files/cms/hhh.yaml if necessary

python3 -m spanet.train -of options_files/cms/training_settings.json

python3 -m spanet.test spanet_output/version_0 -tf data/hhh_testing.h5  --gpu
