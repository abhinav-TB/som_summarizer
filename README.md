# som_summarizer
A method to summarize the text in the English language has been proposed. The proposed method performs extractive summarization by clustering employing a deep neural network known as the self-organizing map.
## Prerequisite
- Python 3.6+ (tested on python 3.8)
- nltk
- sklearn-som
- sentence-transformers
- primefac

## Installation

### Using pip
``` pip install -i https://test.pypi.org/simple/ som-summarizer```

### Building from source


   ```sh
   git clone https://github.com/abhinav-TB/som_summarizer.git
   ```
   ```
   cd text-summarization
   ```
   ```sh
   pip install .
   ```

## Usage
```
from som_summarizer import summarizer

input = ''' ''' #text to be summarized
s = summarizer(sum_size= 4 ,epochs=100) # summary size and epoch size
print(s.generate_summary(input))
```
For  demo run the demo.py file in the root directory
## Local development setup

1.Clone the repository
```
git clone https://github.com/abhinav-TB/som_summarizer.git
```
2.Change directory
```
cd text-summarization
```
3.Install the dependencies

```
pip install -r requirements.txt
```
## Testing

```
pytest .
```
## web app demo
Use [front-end](https://github.com/abhinav-TB/text-summarization/tree/FrontEnd) branch for further installation instructions 
##  License

Distributed under the Apache License. See `LICENSE` for more information.

