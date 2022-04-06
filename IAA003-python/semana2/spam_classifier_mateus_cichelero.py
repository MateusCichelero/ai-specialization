## ALUNO: MATEUS CICHELERO DA SILVA

### O QUE FOI ALTERADO:
# - INCLUSÃO DE FUNÇÕES DE MÉTRICAS DE PERFORMANCE PARA CLASSIFICAÇÃO BINÁRIA DO SCIKIT LEARN
# - MODIFICAÇÃO DA FUNÇÃO tokenizer PARA IDENTIFICAÇÃO DE PALAVRAS QUE CONTÉM NUMEROS COM contains:number
# - ADIÇÃO DE ROTINA NO TREINAMENTO PARA CONSIDERAR EM NOSSO VOCABULÁRIO DE TOKENS SOMENTE PALAVRAS QUE APARECEM 
# UM NÚMERO MINIMO DE VEZES NO DATASET DE TREINAMENTO (função min_count)
# - ADIÇÃO DE FLUXO DE PREPROCESSAMENTO PARA SUBSTITUIÇÃO DE LINKS, EMAILS E CARACTERES ESPECIAIS POR TOKENS
#  ESPECIFICOS (função word_salad)
# - APLICAÇÃO DE STEMMER (SnowballStemmer) E REMOÇÃO DE STOPWORDS USANDO NLTK


# Obs: foi realizada a tentativa de tratar dados do corpo da mensagem (inspirado em
#  https://sdsawtelle.github.io/blog/output/spam-classification-part1-text-processing.html), mas os resultados foram 
# inferiores ao tratar somente o assunto, além de erros nos cálculos das probabilidades da função de predição. 


# Resultado dessa aplicação:
# Comparando com a solução original, observou-se uma melhora
# nos indicadores relacionados a classe positiva (spam), com maior 
# numero de verdadeiros-positivos e melhora no recall da classe spam. 
# Ao mesmo tempo, o número de falsos positivos também aumentou consideravelmente,
# o que impactou basicamente o restante das métricas.
# Matriz Confusão
# [[589  97]
#  [ 41  98]]
# ###
# verdadeiro-negativo: 589
# falso-positivo: 97
# falso-negativo: 41
# verdadeiro-positivo: 98
# ###
#               precision    recall  f1-score   support

#     NAO-SPAM       0.93      0.86      0.90       686
#         SPAM       0.50      0.71      0.59       139

#     accuracy                           0.83       825
#    macro avg       0.72      0.78      0.74       825
# weighted avg       0.86      0.83      0.84       825


################################### IMPORT LIBRARIES ###################################

from typing import TypeVar, List, Tuple, Dict, Iterable, NamedTuple, Set
from collections import defaultdict, Counter
import re
import random
import math
import nltk
from nltk.stem.snowball import SnowballStemmer
import glob


from nltk.corpus import stopwords
nltk.download("stopwords")
X = TypeVar('X')  # generic type to represent a data point

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


################################### DEFINE SOME FUNCTIONS ###################################


def get_metrics(y_true: Iterable[bool], y_pred: Iterable[bool]) -> None:
    '''Print some binary classification metrics
    Inputs:
        - y_true: a list of boolean elements representing the real label: [False, False, True]
        - y_pred: a list of boolean elements representing the predicted result:[True, False, True]
    '''
    print('Confusion Matrix')
    print(confusion_matrix(y_true, y_pred))
    print('###')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f'true-negative: {tn}')
    print(f'false-positive: {fp}')
    print(f'false-negative: {fn}')
    print(f'true-positive: {tp}')
    print('###')

    target_names = ['NON-SPAM', 'SPAM']
    print(classification_report(y_true, y_pred, target_names=target_names))



def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
    """Split data into fractions [prob, 1 - prob]"""
    data = data[:]                    # Make a shallow copy
    random.shuffle(data)              # because shuffle modifies the list.
    cut = int(len(data) * prob)       # Use prob to find a cutoff
    return data[:cut], data[cut:]     # and split the shuffled list there.

def tokenize(text: str) -> Set[str]:
    '''Split input text into tokens. 
    Also converts numbers into a special token contains:number'''
    text = text.lower()                         
    all_words = re.findall("[a-z0-9']+", text)
    all_words = list(map(lambda word: "contains:number" if re.search(r'\d', word) else word, all_words))
    return set(all_words)                       

assert tokenize("Data Science is science") == {"data", "science", "is"}
assert tokenize("Data Science is 1000") == {"contains:number","data", "science", "is"}


def min_count(messages: Iterable[str], minimal_counts:int) -> str:
    '''Takes a list of strings as input and returns the words that appear more than
    minimal_counts times'''
    text = ' '.join(messages).lower()
    counter = Counter(text.split())
    most_common_words = [item[0] for item in counter.items() if item[1] >= minimal_counts]
    return ' '.join(most_common_words)

assert min_count(['a Vaca','o Boi','A galinha', 'o vaca', 'o pato', 'a vaca e o boi', 'o menino'], 2) == 'a vaca o boi'


def word_salad(body):
    '''Produce a word salad from text body.'''  
    all_stopwords = stopwords.words('english')
    
    # lowercase everything
    body = body.lower()
    
    # Replace all URLs with special strings
    regx = re.compile(r"(http|https)://[^\s]*")
    body, nhttps = regx.subn(repl=" httpaddr ", string=body)
    all_stopwords.append('httpaddr')


    # Replace all email addresses with special strings
    regx = re.compile(r"\b[^\s]+@[^\s]+[.][^\s]+\b")
    body, nemails = regx.subn(repl=" emailaddr ", string=body)
    all_stopwords.append('emailaddr')


    # Replace all $, ! and ? with special strings
    regx = re.compile(r"[$]")
    body = regx.sub(repl=" dollar ", string=body)
    regx = re.compile(r"[!]")
    body = regx.sub(repl=" exclammark ", string=body)
    regx = re.compile(r"[?]")
    body = regx.sub(repl=" questmark ", string=body)


    # Remove all other punctuation (replace with white space)
    regx = re.compile(r"([^\w\s]+)|([_-]+)")  
    body = regx.sub(repl=" ", string=body)
    
    # Replace all newlines and blanklines with special strings
    regx = re.compile(r"\n")
    body = regx.sub(repl=" newline ", string=body)
    regx = re.compile(r"\n\n")
    body = regx.sub(repl=" blankline ", string=body)
    all_stopwords.append('newline')
    all_stopwords.append('blankline')

    # Make all white space a single space
    regx = re.compile(r"\s+")
    body = regx.sub(repl=" ", string=body)

    # Remove any trailing or leading white space
    body = body.strip(" ")
 
    # Remove all useless stopwords
    bodywords = body.split(" ")
    keepwords = [word for word in bodywords if word not in all_stopwords]

    # Stem all words
    stemmer = SnowballStemmer("english")
    stemwords = [stemmer.stem(wd) for wd in keepwords]
    body = " ".join(stemwords)

    return body


################################### DEFINE SOME CLASSES ###################################


class Message:    
    def __init__(self, text, is_spam):
        self.text = text
        self.is_spam = is_spam

class NaiveBayesClassifier:
    def __init__(self, k: float = 0.5) -> None:
        self.k = k  # smoothing factor

        self.tokens: Set[str] = set()
        self.token_spam_counts: Dict[str, int] = defaultdict(int)
        self.token_ham_counts: Dict[str, int] = defaultdict(int)
        self.spam_messages = self.ham_messages = 0

    def train(self, messages: Iterable[Message], minimal_counts: int) -> None:

        common_tokens = tokenize(min_count([message.text for message in messages], minimal_counts))
        
        for message in messages:
            # Increment message counts
            if message.is_spam:
                self.spam_messages += 1
            else:
                self.ham_messages += 1
            
            # Increment word counts
            # só adiciono no self.tokens de fizer parte do meu set de tokens comuns
            for token in tokenize(message.text):
                if token in common_tokens:
                    self.tokens.add(token)
                    if message.is_spam:
                        self.token_spam_counts[token] += 1
                    else:
                        self.token_ham_counts[token] += 1

    def probabilities(self, token: str) -> Tuple[float, float]:
        """returns P(token | spam) and P(token | not spam)"""
        spam = self.token_spam_counts[token]
        ham = self.token_ham_counts[token]

        p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
        p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

        return p_token_spam, p_token_ham

    def predict(self, text: str) -> float:
        text_tokens = tokenize(text)
        log_prob_if_spam = log_prob_if_ham = 0.0

        # Iterate through each word in our vocabulary.
        for token in self.tokens:
            prob_if_spam, prob_if_ham = self.probabilities(token)

            # If *token* appears in the message,
            # add the log probability of seeing it;
            if token in text_tokens:
                log_prob_if_spam += math.log(prob_if_spam)
                log_prob_if_ham += math.log(prob_if_ham)

            # otherwise add the log probability of _not_ seeing it
            # which is log(1 - probability of seeing it)
            else:
                log_prob_if_spam += math.log(1.0 - prob_if_spam)
                log_prob_if_ham += math.log(1.0 - prob_if_ham)

        prob_if_spam = math.exp(log_prob_if_spam)
        prob_if_ham = math.exp(log_prob_if_ham)
        return prob_if_spam / (prob_if_spam + prob_if_ham)


def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
    '''Returns the probability of a certain model token being a spam'''
    prob_if_spam, prob_if_ham = model.probabilities(token)

    return prob_if_spam / (prob_if_spam + prob_if_ham)

################################### MODEL TESTING ###################################


messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]

model = NaiveBayesClassifier(k=0.5)
model.train(messages,1)

assert model.tokens == {"spam", "ham", "rules", "hello"}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {"spam": 1, "rules": 1}
assert model.token_ham_counts == {"ham": 2, "rules": 1, "hello": 1}

text = "hello spam"

probs_if_spam = [
    (1 + 0.5) / (1 + 2 * 0.5),      # "spam"  (present)
    1 - (0 + 0.5) / (1 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (1 + 2 * 0.5),  # "rules" (not present)
    (0 + 0.5) / (1 + 2 * 0.5)       # "hello" (present)
]

probs_if_ham = [
    (0 + 0.5) / (2 + 2 * 0.5),      # "spam"  (present)
    1 - (2 + 0.5) / (2 + 2 * 0.5),  # "ham"   (not present)
    1 - (1 + 0.5) / (2 + 2 * 0.5),  # "rules" (not present)
    (1 + 0.5) / (2 + 2 * 0.5),      # "hello" (present)
]

p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))

# Should be about 0.83
assert math.isclose(model.predict(text), p_if_spam / (p_if_spam + p_if_ham))


######################## APPLYING THE MODEL IN THE MAIN FUNCTION #########################

def main():

    # modifiquei para incluir mais um nível de diretórios
    path = 'emails/*/*/*'

    data: List[Message] = []
    
    # glob.glob returns every filename that matches the wildcarded path
    for filename in glob.glob(path):
        is_spam = "ham" not in filename
    
        # There are some garbage characters in the emails, the errors='ignore'
        # skips them instead of raising an exception.
        with open(filename, errors='ignore') as email_file:
            for line in email_file:
                if line.startswith("Subject:"):
                    subject = line.lstrip("Subject: ")
                    data.append(Message(subject, is_spam))
                    break  # done with this file
    
    for message in data:
        message.text = word_salad(message.text)
    
    random.seed(0)      # just so you get the same answers as me
    train_messages, test_messages = split_data(data, 0.75)
    
    model = NaiveBayesClassifier()
    model.train(train_messages, 10)
    
    predictions = [(message, model.predict(message.text))
                   for message in test_messages]
    
    # Assume that spam_probability > 0.5 corresponds to spam prediction
    # and count the combinations of (actual is_spam, predicted is_spam)
    y_true = [message.is_spam for message, spam_probability in predictions]
    y_pred = [spam_probability > 0.5 for message, spam_probability in predictions]
    
    
    get_metrics(y_true, y_pred)
    
    words = sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))
    print("spammiest_words", words[-10:])
    print("hammiest_words", words[:10])


if __name__ == "__main__":
    main()
