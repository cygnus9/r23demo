#!/bin/bash

set -euo pipefail

echo "Making virtual env"
python3 -m venv twinkle --copies
cd twinkle
. bin/activate
pip3 install -r ../requirements.txt

echo "Installing demo"
cd ..
cp transforms.py fbo.py demo.py twinkle/
cp text.gltf fm.mp4 music.ogg twinkle/
mkdir -p twinkle/assembly
cp assembly/*py twinkle/assembly
mkdir -p twinkle/geometry
cp geometry/*py twinkle/geometry
cp demo.py twinkle/__main__.py

echo "Packing demo"
cd twinkle/
zip -q ../twinkle-demo *
cd ..
echo '#!/usr/bin/env python3' > twinkle-demo.linux
cat twinkle-demo.zip >> twinkle-demo.linux
chmod a+x twinkle-demo.linux
rm twinkle-demo.zip

echo "Done"
