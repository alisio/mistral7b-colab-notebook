# Mistral AI Text Generation Notebook

This notebook demonstrates how to use the Mistral AI language model to generate text. The Mistral AI model is a transformer-based language model that has been fine-tuned for text generation tasks.

## Installation

To run this notebook, you will need to install the following packages:

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

## Usage

To use this notebook, first import the necessary packages and install any missing dependencies. Then, create an instance of the CTransformers class and load the Mistral AI model using the from_pretrained method. You can then use the invoke method to generate text based on a given prompt.

## Example

Here is an example of how to use the Mistral AI model to generate text:

```python
from langchain_community.llms import CTransformers

llm = CTransformers(model="mistral")
llm.invoke("Who was ayrton senna?")
```

## Author

This notebook was created by Antonio Alisio de Meneses Cordeiro (alisio.meneses@gmail.com).

## License

This notebook is licensed under the MIT License. See the LICENSE file for more information.

## Acknowledgements

This notebook was adapted from the Mistral AI text generation tutorial by TheBloke.