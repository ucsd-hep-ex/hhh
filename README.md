# hhh

Pull and start the Docker container:
```bash
docker pull jmduarte/hhh
docker run -it jmduarte/hhh bash
```

Check out this GitHub repository:
```bash
cd work
git clone git@github.com:jmduarte/hhh
```

Install the Python package:
```bash
cd hhh
pip install -e .
```

To run the training (WIP):
```bash
python src/models/train_model.py
```

To run the prediction and evaluation metrics (WIP):
```bash
python src/models/predict_model.py
```
