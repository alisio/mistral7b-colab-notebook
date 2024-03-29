![mistral wind](./mistral_wind.png)

# Mistral AI Text Generation Notebook

Why, hello there, citizen data scientists, and curious individuals!


This repository contains two Jupyter notebooks that demonstrates how to interact with the Mistral AI language model to generate text. The notebooks provides two methods for running the model:

1. Running the Model inside Google Colab: This method requires you to install certain packages and dependencies, including langchain, huggingface-hub, hf_transfer, accelerate, numpy, pandas, langchain-core, and langchain-mistralai. Depending on your system, you may also need to install ctransformers with CUDA support.

1. Interact with a locally served Mistral: This method requires you to install the Ollama service and certain Python packages, including langchain-community and langchain-core. You will then use Ollama to download and start the service with the Mistral model.

Both methods allow you to generate text based on a given prompt using the invoke method provided by the langchain class. The notebook provides an example of how to use the Mistral AI model to generate text.

By following the steps outlined in this repository, you can start your LLM journey, easily interact with the Mistral AI language model, and generate text based on a given prompt. Whether you choose to run the model inside Google Colab or interact with a locally served Mistral, this repository provides everything you need to get started.



## Installation

### Notebook: Running the Model inside Google Colab

To run the `mistral7b_colab.ipynb` notebook, you will need to install the following packages:

* `langchain`==0.1.13
* `huggingface`-hub==0.22.0
* `hf_transfer`==0.1.6
* `accelerate`==0.28.0
* `numpy`==1.25.2
* `pandas`==1.5.3
* `langchain`-core
* `langchain`-mistralai

You can install these packages using pip by running the following command in a terminal:

```python
pip install langchain==0.1.13 huggingface-hub==0.22.0 hf_transfer==0.1.6 accelerate==0.28.0 numpy==1.25.2 pandas==1.5.3 langchain-core langchain-mistralai
```

If you are running this notebook on a macOS system, you will also need to install the ctransformers package. You can do this by running the following command in a terminal:

```
CT_METAL=1 pip install ctransformers==0.2.27 --no-binary ctransformers
```

If you are running this notebook on a system with a CUDA-enabled GPU, you can install the ctransformers package with CUDA support by running the following command in a terminal:

```
pip install ctransformers[cuda]==0.2.27
```

### Usage

To use this notebook, first import the necessary packages and install any missing dependencies. Then, create an instance of the class and load the Mistral AI model using the from_pretrained method. You can then use the invoke method to generate text based on a given prompt.

### Example

Here is an example of how to use the Mistral AI model to generate text:

```python
from langchain_community.llms import CTransformers

llm = CTransformers(model="mistral")
llm.invoke("Who was ayrton senna?")
```
### Notebook: Interact with a model served by Ollama

To run the `mistral7b_colab.ipynb` notebook, you will need to install the following packages:

* `langchain`==0.1.13
* `huggingface`-hub==0.22.0
* `hf_transfer`==0.1.6
* `accelerate`==0.28.0
* `numpy`==1.25.2
* `pandas`==1.5.3
* `langchain`-core
* `langchain`-mistralai

You can install these packages using pip by running the following command in a terminal:

```python
pip install langchain==0.1.13 huggingface-hub==0.22.0 hf_transfer==0.1.6 accelerate==0.28.0 numpy==1.25.2 pandas==1.5.3 langchain-core langchain-mistralai
```

If you are running this notebook on a macOS system, you will also need to install the ctransformers package. You can do this by running the following command in a terminal:

```
CT_METAL=1 pip install ctransformers==0.2.27 --no-binary ctransformers
```

If you are running this notebook on a system with a CUDA-enabled GPU, you can install the ctransformers package with CUDA support by running the following command in a terminal:

```
pip install ctransformers[cuda]==0.2.27
```

### Usage

To use this notebook, first import the necessary packages and install any missing dependencies. Then, create an instance of the class and load the Mistral AI model using the from_pretrained method. You can then use the invoke method to generate text based on a given prompt.

### Example

Here is an example of how to use the Mistral AI model to generate text:

```python
from langchain_community.llms import CTransformers

llm = CTransformers(model="mistral")
llm.invoke("Who was ayrton senna?")
```

## Interact with a locally served Mistral

[This notebook](./mistral7b_ollama.ipynb) demonstrates how to interact with a locally served Mistral AI language model to generate text using Ollama.

### Step 1: Prerequisites
To interact with the "Mistral" open-source Large Language Model (LLM) locally through Ollama, follow these steps:

1. Install Ollama Service: Follow the setup instructions on the Ollama GitHub repository here to install and run the Ollama service on your system.

1. Start Ollama: Use Ollama to download and start the service with the Mistral model using ollama run mistral.

1. Install Python Packages: Ensure you have the required Python packages installed to interact with the Ollama service and utilize the Mistral model effectively.

    * `langchain-community`==0.0.29
    * `langchain-core`==0.1.36

You can install these packages using pip:
```python
!pip install -q langchain-community==0.0.29 langchain-core==0.1.36
```

### Step 2: Prepare the Mistral AI Model
Preparing a model served by Ollama is simpler compared to downloading and running it within the Colab environment. Although there are multiple parameters, you only need to set the model name and, optionally, the base URL if you're using a remote Ollama service.

```python
from langchain_community.llms import Ollama

llm = Ollama(model="mistral", base_url="http://localhost:11434")
```

### Step 3: Generate Text Using the Mistral AI Model
In this step, we will use the Mistral AI model to generate text based on a given prompt. We will use the invoke method provided by the langchain class to generate text. We will specify the prompt as an argument to the invoke method and then print the generated text to the console.

```python
%%time

for text in llm.invoke("Write the summary of the book 'The Alchemist', by Paulo Coelho"):
    print(text, end="", flush=True)
```

## References

* Mistral AI: https://huggingface.co/mistralai
* Langchain: https://github.com/mistralai/langchain
* Hugging Face Hub: https://huggingface.co/docs/hub/index
* HF Transfer: https://github.com/huggingface/transfer
* Accelerate: https://github.com/huggingface/accelerate
* NumPy: https://numpy.org/
* Pandas: https://pandas.pydata.org/
* Langchain-Core: https://github.com/mistralai/langchain-core
* Langchain-MistralAI: https://github.com/mistralai/langchain-mistralai
* Ollama: https://github.com/ollama/ollama
* ctransformers: https://github.com/huggingface/ctransformers


## Author

This notebook was created by Antonio Alisio de Meneses Cordeiro (alisio.meneses@gmail.com).

## License

This notebook is licensed under the MIT License. See the LICENSE file for more information.

## Acknowledgements

This notebook was adapted from the Mistral AI text generation tutorial by TheBloke.
