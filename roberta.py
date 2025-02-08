import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import warnings
warnings.filterwarnings('ignore')

# Fix all the seeds
def seed_everything(seed: int) -> None:
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(42)

MODEL_NAME = 'FacebookAI/roberta-large' # 'microsoft/deberta-v3-large' #
DIR_NAME = 'roberta-best' #'deberta-best' #
FULL_DIR = 'finetune_dir/'+DIR_NAME
MAX_LEN = 200
TRAIN_BATCH_SIZE = 32
EPOCHS = 10
number = 0
DATA = ''
FILE_NAME = f'pred_eng_a_{DIR_NAME}_{EPOCHS}ep{DATA}.csv'
THRESHOLD = 0.2
LEARNING_RATE = 2e-5
LOG_STEP = 200
pd.set_option("display.max_columns", None)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(FULL_DIR):
    os.makedirs(FULL_DIR)
    print(f'---> Created [{FULL_DIR}] directory\n')


class BERTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, target_cols):
        self.df = df
        self.max_len = max_len
        self.text = df.text
        self.tokenizer = tokenizer
        self.targets = df[target_cols].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        input_ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(self.targets[index], dtype=torch.float)
        }


df_train = pd.read_csv("data/train_eng_a.csv")
if DATA != '':
    df_train_2 = pd.read_csv(f"data/train_{DATA}_a.csv")
    df_train = df_train.sample(frac=1, random_state=42, ignore_index=True)
dev_out = pd.read_csv("data/dev_eng_a.csv")

print(df_train.head())
print('\nTrain len:', df_train.shape)

target_cols = [col for col in df_train.columns if col not in ['text', 'id', 'Disgust']]
print('target_cols:', target_cols)

print(f'\nLoading {MODEL_NAME}\n')

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(target_cols))
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = BERTDataset(df_train, tokenizer, MAX_LEN, target_cols)
# valid_dataset = BERTDataset(df_dev, tokenizer, MAX_LEN, target_cols)
out_dataset = BERTDataset(dev_out, tokenizer, MAX_LEN, target_cols)

args = TrainingArguments(output_dir="logs/model",
                         num_train_epochs=EPOCHS,
                         per_device_train_batch_size=TRAIN_BATCH_SIZE,
                         save_steps=10000000,
                         logging_steps=LOG_STEP,
                         save_total_limit=2,
                         weight_decay=1e-6,
                         learning_rate=LEARNING_RATE,
                        #  fp16=True,
                         )

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    args=args,
)

print(f'->>> Start training with the following parameters:')
print(f'num_train_epochs {EPOCHS},\nper_device_train_batch_size {TRAIN_BATCH_SIZE},\n\n')

trainer.train()

print('\nDone with training.\n')

saved_name = FILE_NAME.replace('pred_eng_a_','')
saved_name = FILE_NAME.replace('.csv','')
dir = FULL_DIR+'/'+saved_name

print(f'>>>> Save results into {dir}\n')
trainer.save_model(dir)
print(f'>>>> Trainer saved model\n')

for t in range(1,7):
    if t == 1:
        t = 1.5
    threshold = t*0.1
    print(f'\n----> Threshold {threshold}\n')
    outputs = trainer.predict(out_dataset)
    outputs = np.array(torch.sigmoid(torch.tensor(outputs.predictions))) >= threshold
    outputs = outputs.astype(int)
    dev_out[target_cols] = outputs

    print('\n', dev_out.head())

    if t == 1.5: t = 15
    FFILE_NAME = FILE_NAME.replace('.csv',f'_{int(t)}tr.csv')

    index = 0
    while os.path.exists(f'{FULL_DIR}/{FFILE_NAME}'):
        print(f'---> File [{FULL_DIR}/{FFILE_NAME}] exists\n')
        if index == 0:
            FFILE_NAME = FFILE_NAME.split('.csv')[0] + '_' + str(index) + '.csv'
        else:
            FFILE_NAME = FFILE_NAME.split('.csv')[0][:-2] + '_' + str(index) + '.csv'
        index += 1

    dev_out.to_csv(f'{FULL_DIR}/{FFILE_NAME}', header=['id', 'Anger', 'Fear', 'Joy', 'Sadness', 'Surprise'],
                columns=['id', 'Anger', 'Fear', 'Joy', 'Sadness', 'Surprise'], index=False)


