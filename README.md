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

## Additional experiments on reward functions
We opt for a simple reward function for two main reasons. First, this reward design sufficiently supports the agent in learning effective policies and facilitates fair comparisons with existing methods. Second, we aim to minimize task-specific customizations to maintain the model's generalizability; overly complex or inaccurate rewards could diminish the model's performance.

| Methods | Validation Seen | Validation Unseen |
|-------------- |---------------------------- |------- |
| | NL↓ NE↓ SR↑ SPL↑ |NL↓ NE↓ SR↑ SPL↑ |
|DILLM-VLN| 12.8 4.74 57.2 0.51 | 11.4 5.31 49.4 0.44 |
| + SGS| 12.3 5.15 53.5 0.48 | 12.8 5.37 47.5 0.41 |
| + OGS| 11.8 5.27 52.2 0.47 |11.7 5.66 46.1 0.40 |

We have incorporated the scene grounding score (SGS, which assesses if the agent has reached the scene described by the sub-instruction) and the object grounding score (OGS, which determines if the agent has found the target object described in the sub-instruction) into the reward function. The above table presents the experiment results, showing a decline in navigation performance with the addition of SGS and OGS. This indicates that the design of the reward function directly influences the learning objectives of the agent. Our task design already decomposes the navigation task into multiple simple sub-instruction, focusing the agent on completing each sub-instruction sequentially. The additional reward signals introduce unnecessary distractions, hindering the agent's learning of efficient navigation policies.