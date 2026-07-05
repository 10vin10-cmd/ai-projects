# ai_toolbox

A small collection of helpful AI-related utilities.

## Installation

Install from PyPI (when published):

```
pip install ai_toolbox
```

## Quick start

Use the library from Python:

```py
from ai_toolbox import add

print(add(2, 3))  # -> 5
```

Or use the CLI after installing the package:

```
ai-toolbox add 2 3
```

See `docs/index.md` for more details.

Notes on logs for Dependency: 
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
ai-toolbox 0.1.0 requires click==8.2.1, but you have click 8.1.8 which is incompatible.
ai-toolbox 0.1.0 requires flask==2.2.2, but you have flask 3.1.3 which is incompatible.
ai-toolbox 0.1.0 requires litellm==1.74.14, but you have litellm 1.83.7 which is incompatible.
ai-toolbox 0.1.0 requires python-dotenv==1.1.1, but you have python-dotenv 1.0.1 which is incompatible.

For RAG:
!pip install ibm-watsonx-ai==0.2.6
!pip install langchain==0.1.16
!pip install langchain-ibm==0.1.4
!pip install transformers==4.41.2
!pip install huggingface-hub==0.23.4
!pip install sentence-transformers==2.5.1
!pip install chromadb
!pip install wget==3.2
!pip install --upgrade torch --index-url https://download.pytorch.org/whl/cpu