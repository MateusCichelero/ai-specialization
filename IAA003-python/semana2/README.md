Curso de Especialização de Inteligência Artificial Aplicada

Setor de Educação Profissional e Tecnológica - SEPT

Universidade Federal do Paraná - UFPR

---

**IAA003 - Linguagem de Programação Aplicada**

Prof. Alexander Robert Kutzke
Aluno Mateus Cichelero da Silva

# Exercício de implementação do algoritmo Naive Bayes

O QUE FOI ALTERADO:
- INCLUSÃO DE FUNÇÕES DE MÉTRICAS DE PERFORMANCE PARA CLASSIFICAÇÃO BINÁRIA DO SCIKIT LEARN
- MODIFICAÇÃO DA FUNÇÃO tokenizer PARA IDENTIFICAÇÃO DE PALAVRAS QUE CONTÉM NUMEROS COM contains:number
- ADIÇÃO DE ROTINA NO TREINAMENTO PARA CONSIDERAR EM NOSSO VOCABULÁRIO DE TOKENS SOMENTE PALAVRAS QUE APARECEM UM NÚMERO MINIMO DE VEZES NO DATASET DE TREINAMENTO (função min_count)
- ADIÇÃO DE FLUXO DE PREPROCESSAMENTO PARA SUBSTITUIÇÃO DE LINKS, EMAILS E CARACTERES ESPECIAIS POR TOKENS ESPECIFICOS (função word_salad)
- APLICAÇÃO DE STEMMER (SnowballStemmer) E REMOÇÃO DE STOPWORDS USANDO NLTK


Obs: foi realizada a tentativa de tratar dados do corpo da mensagem (inspirado em https://sdsawtelle.github.io/blog/output/spam-classification-part1-text-processing.html), mas os resultados foram inferiores ao tratar somente o assunto, além de erros nos cálculos das probabilidades da função de predição. 
