# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 20:23:21 2023

@author: nikch
"""
import itertools
def choose_sets(lst, k):
    ''' Input: lst,
        Output: a list of all k-length sub-lists '''
    if len(lst)==k :
        return [lst]
    if  k==0:
        return [[]]
    if k==1:
        return [[i] for i in lst]
    sub_lst1=choose_sets(lst[1:],k-1)
    for i in sub_lst1:
        i.append(lst[0])
    sub_lst2=choose_sets(lst[1:],k)
    final_lst=[]
    final_lst.extend(sub_lst1)
    final_lst.extend(sub_lst2)
    return final_lst

def main():
    
    # Чтение вводных данных
    n = int(input())
    decks = []

    for _ in range(n):
        size = int(input())
        cards = list(map(int,input().split()))
        decks.append(cards)
        print(choose_sets(cards,4))
        print(len(choose_sets(cards,4)))
        cur = 0 
        for x in choose_sets(cards,4):
            cur += sum(sorted(x,reverse = True)[:3])
        print(cur/len(choose_sets(cards,4)))
    # Вычисление математического ожидания для каждой колоды
    max_expectation = float('-inf')
    best_deck = -1

    for i, deck in enumerate(decks):
        deck.sort(reverse=True)
        expectation = sum(deck[:3])
        
        if expectation > max_expectation:
            max_expectation = expectation
            best_deck = i + 1
    
    # Вывод результатов
    print(best_deck, round(max_expectation / len(decks), 4))

if __name__ == "__main__":
    main()
"""
from itertools import combinations

# Функция для вычисления математического ожидания
def expected_value(nums):
    return sum(sorted(nums, reverse=True)[:3])

# Считываем количество колод
n = int(input())

# Список колод и их математических ожиданий
decks = []
expectations = []

def calc_expectation(values):
    prob = list(values)
    res = 0
    for i in range(len(prob)):
        prob[i] = 1 / len(values)
        res += values[i] * prob[i]
    return res
# Считываем информацию о каждой колоде
for _ in range(n):
    size = int(input())
    cards = list(map(int, input().split()))
    decks.append(cards)
    expectations.append(0)

print(calc_expectation(decks[0]))
# Находим колоду с наибольшим математическим ожиданием
max_expectation = max(expectations)
max_deck = decks[expectations.index(max_expectation)]

# Выводим результат
print(max_expectation)
print(*max_deck)
"""