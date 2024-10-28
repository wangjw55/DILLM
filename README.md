## Environment Installation

### Setup:

#### 1. Use [anaconda](https://anaconda.org/) to create a Python 3.8 environment:
```bash
conda create -n vln python3.8
conda activate vln
```

#### 2. Install [CLIP](https://github.com/openai/CLIP):
```bash
pip install git+https://github.com/openai/CLIP.git
```

#### 3. Install dependencies:
```bash
pip install -r python_requirements.txt
```

#### 4. Install [Matterport3D](https://github.com/peteanderson80/Matterport3DSimulator) simulators (v0.1):
```bash
sudo apt-get install libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev
mkdir build && cd build
cmake -DEGL_RENDERING=ON ..
make -j8
```

#### 5. Download image and object features for environments, unzip and place under img_features/:

- Please check [here](https://drive.google.com/file/d/1GBtMB0HINkwInyj-i0wLYBM-lSVJq6lj/view?usp=drive_link) for image features
- Please check [here](https://drive.google.com/file/d/1k9n4Js0dA8zxl3cV9fGvCHS8cAri3Pst/view?usp=drive_link) for object features

#### 6. Download the [CLIP](https://github.com/openai/CLIP) model (optional)

Download the CLIP model [here](https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt) and place it under img_features/ or run the script to automatically download the model

#### 7. Download [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B) model (optional)
Download the ChatGLM-6B model [here](https://huggingface.co/THUDM/chatglm3-6b/tree/main) for online instruction decomposition, but this is not necessary because the instructions have been pre-processed and stored in the json file under tasks/R2R/data/
## Code

### Agent
```
bash run/agent.bash 0
```
0 is the id of GPU. It will train the agent and save the snapshot under snap/agent/.

### Test Agent
After training the agent, test the agent by
```
bash run/test_agent.bash 0
```
0 is the id of GPU. 
It will load trained agent and test it on test set.