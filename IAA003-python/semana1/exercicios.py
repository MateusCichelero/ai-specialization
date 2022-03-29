#1 Escreva uma função que converte horas, minutos e segundos em um número total de segundos.
def convert_to_seconds(hours, minutes, seconds):
    time_in_seconds = hours*3600 + minutes*60 + seconds

    return time_in_seconds

def is_palindrome(text):
    return(text == text[::-1])
 

def soma_de_quadrados(xs):
    return sum([x*x for x in xs]) 


def recursive_factorial(num):
    if num == 1:
        return num
    else:
        return  num * recursive_factorial(num-1)

def recursive_list_invertion(a_list):
    if a_list:
        return([a_list[-1]] + recursive_list_invertion(a_list[0:-1]))
    else:
        return([])


#Utilize List Comprehension para elevar ao quadrado todos os elementos de uma lista;
def square_list_elements(xs):
    return [x*x for x in xs] 


#Utilize List Comprehension para retirar elementos palíndromos de uma lista de strings;
lista_strings = ['teste', 'ana', 'urubu', 'lol']

[x for x in lista_strings if not is_palindrome(x)]


#Utilize List Comprehension para, dadas as listas A e B, criar uma lista C composta apenas pelos elementos presentes em ambas A e B;
a = [1,2,3, 10,20,30,22, 55] 
b = [2,4,5, 55, 30]

[x for x in a if x in b]

[x for x in a if x not in b] or [x for x in b if x not in a]

#ou usando sets
[x for x in set(a+b) if x in a and x in b]