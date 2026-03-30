#!/bin/sh

python -m venv env
source env/bin/activate
pip install maturin
maturin develop --release

echo "BUILD COMPLETE"

python -c "from dynamo import ProbPos, DetPos, Module, GeneralModule, Construct; print('Rust OK')"
python -c "from dynamo.rp import parameters; print('RP OK')"
python -c "from dynamo import vis; print('Vis OK')"
