# som_summarizer
## Prerequisite
Python 3.6+ (tested on python 3.8)

## Installation

### Using pip
``` pip install -i https://test.pypi.org/simple/ som-summarizer```

### Building from source


   ```sh
   git clone https://github.com/abhinav-TB/Time-Series-Forecasting-Uisng-LSTM.git
   ```
   ```sh
   pip install .
   ```

## Usage
```
from som_summarizer import summarizer

input = "" # text to be summarized
s = summarizer(epochs=100 ) # epochs for som training
print(s.generate_summary(input))
```
For  demo run the demo.py file in the root directory
## Local development setup

1.Clone the repository
```
   git clone https://github.com/abhinav-TB/Time-Series-Forecasting-Uisng-LSTM.git
```
2.Install the dependencies

```
   pip install -r requirements.txt
```
3. cd to install directory
4. Run 'streamlit run FrontEnd.py'
