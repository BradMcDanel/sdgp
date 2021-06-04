## Getting Started

- Clone the repository to your computer using `git clone git@github.com:BradMcDanel/blocked_stochastic_gradient.git`. Then, enter the folder with `cd blocked_stochastic_gradient`.
- Create a new python virtual environment (venv). This makes it so the python packages are consistent across multiple computers/users. To create the virtual environment, do: `python3 -m venv venv`. This creates a folder called `venv` in your current working directory.
- To activate the virtual environment, run `source venv/bin/activate`. Now, if you run `python`, it will use your newly created instance.
- Next, install all required library dependencies with `pip -r requirements.txt` (this will take several minutes). Now, we have our python environment setup and are ready to start running code. Restart your machine to ensure that PyTorch was installed correctly.

## Training a Neural Network
Run: `python train_cifar10.py cfg/resnet18_cifar10.py` to train the CIFAR-10 network. This training would be quite slow on a CPU, but you should be able to start training and see the loss decrease over time.
