# Muhammad Abdullah
# COM307
# SEPTEMBER 26th 2026

from random import *


# 50% chance of 6 and 10% chance for each of (1,2,3,4,5)
def die_throw():
    num = random()
    if num <= 0.1:
        die_value = 1
    elif num <= 0.2:
        die_value = 2
    elif num <= 0.3:
        die_value = 3
    elif num <= 0.4:
        die_value = 4
    elif num <= 0.5:
        die_value = 5
    else:
        die_value = 6


    return die_value


def generate_dataset(n):
    file_name = f"dataset_{n}rolls.txt"
    rolls_list = []

    for i in range(n):
        rolls_list.append(die_throw())

    with open(file_name, "w", encoding="utf-8") as file:
        
        file.write(str(rolls_list))



def generate_all_datasets(n_list):
    for n in n_list:
        generate_dataset(n)


#generate_all_datasets([20,200,2000])


