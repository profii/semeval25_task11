import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer#, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

from tqdm import tqdm
from collections import defaultdict
import pickle
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
os.environ["WANDB_PROJECT"] = 'semeval25_task11_hatt'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


MODEL_NAMES = ['FacebookAI/roberta-large']
DIR_NAMES = ['roberta']
LANGS = ['eng',
         'arq', 'ary', # Arabic
         'pcm', 'ptbr', 'ptmz', 'ron', 'swa', 'swe', 'deu', 'esp', # Latin
         'tat', 'ukr', 'rus',] # Slavic

MODEL_ID = 0
LANG_ID = 0
is_test = True
THRESHOLD = 0.1

is_sorted = False # False
is_pure = False # True
EPOCHS = 10

LANG = LANGS[LANG_ID]
MODEL_NAME = MODEL_NAMES[MODEL_ID]
DIR_NAME = DIR_NAMES[MODEL_ID]
FILE_NAME = f'pred_{LANG}_{DIR_NAME}_{EPOCHS}ep.csv'
if is_pure:
    FILE_NAME = FILE_NAME.replace('.csv', '_pu.csv')
    DIR_NAME = DIR_NAME+'_pu'
if is_sorted:
    FILE_NAME = FILE_NAME.replace('.csv', '_so.csv')
    DIR_NAME = DIR_NAME+'_so'

if is_test:
    DIR_NAME = DIR_NAME + '_hatt'
    FULL_DIR = f'finetune_dir/test/{LANG}/'+DIR_NAME
else:
    FULL_DIR = f'finetune_dir/{LANG}/'+DIR_NAME
MAX_LEN = 200
TRAIN_BATCH_SIZE = 32
# VALID_BATCH_SIZE = 32
# THRESHOLD = 0.5
LEARNING_RATE = 2e-5
LOG_STEP = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(FULL_DIR):
    os.makedirs(FULL_DIR)
    print(f'---> Created [{FULL_DIR}] directory\n')


class HierarchicalAttention(nn.Module):
    def __init__(self, hidden_dim, num_emotions):
        super(HierarchicalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_emotions = num_emotions
        self.attention_weights_layer = nn.Linear(hidden_dim + 1, 1)

    def forward(self, token_embeddings, emotion_scores, attention_mask):
        # Concatenate token embeddings with emotion scores
        emotion_scores = emotion_scores.unsqueeze(-1)  # [batch_size, seq_len, num_emotions, 1]
        repeated_embeddings = token_embeddings.unsqueeze(-2).repeat(1, 1, self.num_emotions, 1) # [batch_size, seq_len, num_emotions, hidden_dim]
        concatenated = torch.cat([repeated_embeddings, emotion_scores], dim=-1)  # [batch_size, seq_len, num_emotions, hidden_dim + 1]

        # Compute attention logits
        attention_logits = self.attention_weights_layer(concatenated).squeeze(-1)  # [batch_size, seq_len, num_emotions]
        attention_logits = F.softmax(attention_logits, dim=1)  # Normalize across sequence length

        # Mask out padding tokens
        attention_logits = attention_logits * attention_mask.unsqueeze(-1)

        # Weighted sum of token embeddings
        weighted_embeddings = (attention_logits.unsqueeze(-1) * token_embeddings.unsqueeze(-2)).sum(dim=1)

        return weighted_embeddings


class EmotionalRobertaWithAttention(nn.Module):
    def __init__(self, model_name, num_emotions, hidden_dim):
        super(EmotionalRobertaWithAttention, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.attention = HierarchicalAttention(hidden_dim, num_emotions)
        self.classifier = nn.Linear(hidden_dim * num_emotions, num_emotions)

    def forward(self, input_ids, attention_mask, token_emotion_scores, labels=None):
        """
        Forward pass with hierarchical attention.
        """
        # Step 1: Get token embeddings from base model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]

        # Step 2: Apply hierarchical attention
        weighted_embeddings = self.attention(token_embeddings, token_emotion_scores, attention_mask)

        # Step 3: Flatten and pass to the classifier
        logits = self.classifier(weighted_embeddings.view(weighted_embeddings.size(0), -1))

        # Step 4: Compute loss if labels are provided (using BCEWithLogitsLoss for multi-label)
        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)

        return {"logits": logits, "loss": loss} if labels is not None else {"logits": logits}


class BERTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, target_cols, emotional_tokens):
        self.df = df
        self.max_len = max_len
        self.text = df.text
        self.tokenizer = tokenizer
        self.targets = df[target_cols].values
        self.emotional_tokens = emotional_tokens
        self.num_emotions = len(target_cols)

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

        # Generate token-level emotion scores
        token_emotion_scores = [
            self.emotional_tokens.get(self.tokenizer.convert_ids_to_tokens(token_id), [0] * self.num_emotions)
            for token_id in input_ids
        ]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(self.targets[index], dtype=torch.float),
            'token_emotion_scores': torch.tensor(token_emotion_scores, dtype=torch.float)
        }

def emotion_spelling(df, emotions):
    new_list = []
    for idx in range(df.shape[0]):
        letters = ''
        for emotion in emotions:
            if df.iloc[idx][emotion] == 1:
                letters += emotion+','
        if letters == '':
            letters = 'neutral'
        else:
            letters = letters[:-1]
        
        new_list.append(letters)

    df['answer'] = new_list
    print('\nTrain data with emotions:\n', df.head())

    return df

def get_emotional_tokens(df, tokenizer, emotions):
    # Step 1: Tokenize the texts and group tokens by emotion
    token_emotion_count = defaultdict(lambda: defaultdict(int))  # Token -> Emotion -> Count
    emotion_counts = defaultdict(int)  # Overall count of texts for each emotion

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        tokens = tokenizer.tokenize(row["text"])
        labels = row["answer"].split(",")
        # print(_, labels)
        for emotion in labels:
            if emotion != 'neutral':
                emotion_counts[emotion] += 1
                for token in tokens:
                    token_emotion_count[token][emotion] += 1

    # Step 2: Compute probabilities for each token across emotions
    emotional_tokens = {}

    for token, counts in token_emotion_count.items():
        total_count = sum(counts.values())  # Total occurrences of the token
        token_scores = []
        for emotion in emotions:
            # Compute probability: (token count in emotion class / total token count)
            probability = counts[emotion] / total_count if total_count > 0 else 0
            token_scores.append(probability)
        emotional_tokens[token] = token_scores #dict(zip(emotions, token_scores))

    return emotional_tokens



def main():
    train_name = f"../data/train_{LANG}_a.csv"
    if is_test:
        test_name = f"../langs/data/test/{LANG}.csv"
    else:
        dev_name = f"../data/dev_{LANG}_a.csv"

    print('\ntrain_name:',train_name)
    df_train = pd.read_csv(train_name)
    if is_test:
        dev_out = pd.read_csv(test_name)
    else:
        dev_out = pd.read_csv(dev_name)

    target_cols = [col for col in dev_out.columns if col not in ["text", "id"]]
    print('target_cols:\n', target_cols)

    print(df_train.head())
    print('\nTrain len:', df_train.shape)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = df_train.copy(deep=True)
    df = emotion_spelling(df, list(target_cols))
    emotional_tokens = get_emotional_tokens(df, tokenizer, list(target_cols))
    print('emotional_tokens:\n', list(emotional_tokens.items())[:5])

    print(f'\n---> Loading {MODEL_NAME}\n')

    model = EmotionalRobertaWithAttention(
        model_name=MODEL_NAME,
        num_emotions=len(target_cols),
        hidden_dim=1024,  # Roberta Large hidden dimension
    ).to(device)

    print(f'\n---> Model Achitecture <---\n\n')
    print(model)
    print(f'---> Model Achitecture <---\n')

    train_dataset = BERTDataset(df_train, tokenizer, MAX_LEN, target_cols, emotional_tokens)
    out_dataset = BERTDataset(dev_out, tokenizer, MAX_LEN, target_cols, emotional_tokens)

    run_name = FILE_NAME.replace(f'pred_{LANG}_','')
    run_name = run_name.replace('.csv','')

    args = TrainingArguments(
        output_dir="logs/model",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        save_steps=10000000,
        logging_steps=LOG_STEP,
        save_total_limit=2,
        weight_decay=1e-6,
        report_to="wandb",
        run_name=run_name,
        learning_rate=LEARNING_RATE,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        args=args
    )

    print(f'->>> Start training with the following parameters:')
    print(f'num_train_epochs {EPOCHS},\nper_device_train_batch_size {TRAIN_BATCH_SIZE},\n\n')
    trainer.train()

    print('\nDone with training.\n')

    more_dir = str(EPOCHS)+'ep'
    dir = FULL_DIR+'/'+more_dir+'/'+run_name

    print(f'>>>> Save results into {dir}\n')
    trainer.save_model(dir)
    print(f'>>>> Trainer saved model\n')

    print(f'\n----> Threshold {THRESHOLD}\n')
    outputs = trainer.predict(out_dataset)
    outputs = np.array(torch.sigmoid(torch.tensor(outputs.predictions))) >= THRESHOLD
    outputs = outputs.astype(int)
    dev_out[target_cols] = outputs

    print('\n', dev_out.head())

    threshold = THRESHOLD*10
    # FFILE_NAME = FILE_NAME.replace('.csv',f'_{int(threshold)}tr.csv')
    temp_dir = FULL_DIR+'/'+more_dir+f'/{int(threshold)}tr'
    os.makedirs(temp_dir)
    
    FFILE_NAME = temp_dir+f'/pred_{LANG}.csv'
    index = 0
    while os.path.exists(f'{FFILE_NAME}'):
        print(f'---> File [{FFILE_NAME}] exists\n')
        if index == 0:
            FFILE_NAME = FFILE_NAME.split('.csv')[0] + '_' + str(index) + '.csv'
        else:
            FFILE_NAME = FFILE_NAME.split('.csv')[0][:-2] + '_' + str(index) + '.csv'
        index += 1

    colus = ['id']
    colus.extend(target_cols)
    if LANG == 'eng':
        dev_out.to_csv(f'{FFILE_NAME}', header=['id', 'Anger', 'Fear', 'Joy', 'Sadness', 'Surprise'],
                columns=colus, index=False)
    else:
        dev_out.to_csv(f'{FFILE_NAME}', header=['id', 'Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise'],
                columns=colus, index=False)


if __name__ == "__main__":
    main()

