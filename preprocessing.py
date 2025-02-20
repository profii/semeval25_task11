import re
import emoji

list_em = ['Anger','Fear','Joy','Sadness','Surprise']
standard_emoticons = [':@', 'D:', ':)', ':(', ':o', ':|']
emoji_re = [
    ['🤦‍♂️','🤢','😠','😤','🤡','🤬','😡',':-@', '@-:', '@:'],
    ['🫣','D-:',],
    ['🤗','😍','😆','😁','🔥','😊','😂','🤣','😜','😝','😎','😉','😙','😌','😃','😅', ':P', ':-)', ':]', '=)', '(:', '(-:', '[:', '(=', '; )', '( ;', '( :', ': )'],
    ['😢','😭','🥺','😔',':-(', ':[', '=(', '):', ')-:', ']:', ')='],
    ['😱','😳',':-o', 'O_O', 'o_O', 'O:',],
    ['🤨','😐',':-|', ':l', ':I', '|:', '|-:', 'l:', 'I:',],
]

def replace_similar_emojis(text):
    for idx, em in enumerate(list_em):
        for emoji in emoji_re[idx]:
            text = re.sub(re.escape(emoji), standard_emoticons[idx], text)
    return text

def remove_emojis(text):
    return ''.join(char for char in text if char not in emoji.EMOJI_DATA)


def clean_text(text):
    # Replace similar emojis with standard versions
    text = replace_similar_emojis(text)
    text = result = re.sub(r'\._\.', ':|', text)
    text = re.sub(r'[♥💙🖤❤💚]', '*love*', text)
    text = re.sub(r'[🇮🇱🇵🇸]', '', text)
    # Remove other emojis
    text = remove_emojis(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags and other replacements but keep the text
    text = re.sub(r'& # x 26;', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'nbsp;', '', text)
    text = re.sub(r'&lt;3', '*love*', text)
    text = re.sub(r'&lt;', '', text)
    text = re.sub(r'&gt;', '', text)
    text = re.sub(r'&amp;', '', text)
    # Speling
    text = re.sub(r"can't", 'cannot', text)
    text = re.sub(r"I'm", 'I am', text)
    text = re.sub(r"i'm", 'i am', text)
    text = re.sub(r"I'll", 'I will', text)
    text = re.sub(r"i'll", 'i will', text)
    text = re.sub(r" gon na ", " gonna ", text)
    text = re.sub(r" gon ta ", " gotta ", text)
    text = re.sub(r" wan na ", " wanna ", text)
    text = re.sub(r" do nt ", " do not ", text)
    # Remove special characters but keep standard emoticons and punctuation
    text = re.sub(r'(?![:@|D:|:\)|:\(|:o|:\|])[\{\}\$\~\<\>\%\&\=\:\[\]\-]', '', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\`\`', '\"', text)
    text = re.sub(r'\'\'', '\"', text)
    text = re.sub(r'\"\"', '\"', text)
    text = re.sub(r'\`', '\'', text)
    text = re.sub(r'\:\:', '\.', text)
    text = re.sub(r'\.\.', '...', text)
    text = re.sub(r'\.{4,}', '...', text)

    return text

# Example usage
sample_text = "Hello world! `` :-@ & # x 26; 😊"" :P .. 😭 ""mom W....♥ !!!🐑 :o = & %!🇮🇱 {{._.{ Check &lt;3 this: ) ; I'm  nbsp; = / out: ! https://example.com #Exciting @user :["
cleaned_text = clean_text(sample_text)

print("Original Text:", sample_text)
print("Cleaned Text:", cleaned_text)
