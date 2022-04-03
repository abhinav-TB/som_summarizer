from som_summarizer import summarizer
import datasets
import nltk
from rouge import Rouge
import wandb
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

dataset = datasets.load_dataset('ccdv/cnn_dailymail', '3.0.0')

df_train = dataset['train']
df_test = dataset['test']

print(df_test[0])

# wandb.init(project="som-summarizer", entity="abhigamez")


def calculate_score(epochs):
    r = Rouge()
    rouge1 = 0
    rouge2 = 0
    rougeL = 0
    s = summarizer(100)
    for i in range(0,10):
        print(f"set {i}")
        text = df_test[i]['article']
        ref = df_test[i]['highlights']
        summary = s.generate_summary(text)
        # original = s.org_tokens
        # original = "".join(original)
        r_scores = r.get_scores(summary, ref, avg=True)
        rouge1 += r_scores['rouge-1']['f']
        rouge2 += r_scores['rouge-2']['f']
        rougeL += r_scores['rouge-l']['f']

    print(f"Average rouge scores :\n rouge-1 - {rouge1/10}\n rouge2 - {rouge2/10}\n RougeL - {rougeL/10}")
    wandb.log({
        "epochs": epochs,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL
    })

for i in range(20,201,20):
      wandb.init(
      # Set the project where this run will be logged
      project="som-summarizer", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"run for epoch_{i}", 
      # Track hyperparameters and run metadata
      config={
      "dataset": "CNN-dailymail",
      "epcohs": i,
      })
      calculate_score(20)