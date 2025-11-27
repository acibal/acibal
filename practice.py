print("Hello World")

print("Hello AI World")

################################################################################
# Sayılar (Numbers) ve Karakter Dizileri (Strings)
################################################################################

print(9)
print(9.2)

type(9)
type(9.2)
type("Mrb")

################################################################################
# Atamalar ve Değişlenler (Assignments & Variables)
################################################################################

a = 9
a

b = "hello ai era"
b

c = 10

a * c

a * 10

d = a - c

################################################################################
# Virtual Environment (Sanal Ortam) ve (Package Manament) Paket Yönetimi
################################################################################

# Sanal ortamların listelenmesi:
# conda env list

# Sanal ortam oluşturma:
# conda create -n myenv

# Sanal ortamı silme:
# conda remove -n myenv

# Sanal ortamı aktif etme:
# conda activate myenv

# Sanal ortamı deaktif etme:
# conda deactivate myenv

# Yüklü paketlerin listelenmesi:
# conda list

# Paket yükleme:
# conda install numpy

# Aynı anda birden fazla paket yükleme
# conda install pandas scipy

# Paket silme:
# conda remove numpy

# Versiyona göre paket yükleme:
# conda install numpy=1.26.3

# Paket güncelleme:
# conda upgrade numpy

# Tüm paketleri güncelleme:
# conda upgrade --all

##########################################################
# pip: pypi (python packace index) ile paket yükleme:
# pip install pandas

# pip versiyona göre paket yükleme:
# pip install pandas==2.2.1

# pip ile paket silme:
# pip uninstall pandas

# pip ile sanal ortamı dışarı aktarma:
# pip freeze > requirement.txt

##########################################################

# Sanal ortamı dışarı aktarma:
# conda env export > environment.yaml

# Dosya dizinini görme:
# ls

# Dosyadan environment yükleme:
# conda env create -f environment.yaml

################################################################################
# VERİ YAPILARI (DATA STRUCTURES)
################################################################################

# - Veri Yapılarına Giriş ve Hızlı Özet
# - Sayılar (Numbers): int, float, complex
# - Karakter Dizileri (Strings): str
# - Boolean (TRUE-FALSE): bool
# - Liste (List)
# - Sözlük (Dictionary)
# - Demet (Tuple)
# - <<küme (Set)

################################################################################
# Veri Yapılarına Giriş ve Hızlı Özet
################################################################################

# Sayılar: integer

x = 46
type(x)

# Sayılar: float

x = 10.3
type(x)

# Sayılar: complex

x = 2j + 1
type(x)

# String

x = "Hello AI Era"
type(x)

# Boolean

True
False
type(True)

5 == 4
3 == 2
1 == 1
2 != 2
3 != 4

# Liste (List)

x = ["btc", "eth", "xrp"]
type(x)

# Sözlük (Dictionary)

x = {"Name": "Peter",
     "Afe": 36}
type(x)

# Tuple

x = ("python", "ml", "ds")
type(x)

# Set

x = {"python", "ml", "ds"}
type(x)

# List, Dictionary, Tuple, Set (Python Collections) (Python Arrays)

################################################################################
# Sayılar (Numbers): int, float, complex
################################################################################

a = 5
b = 10.5

a * 3
a / 7
a * b / 10
a ** 2

################################################################################
# Tipleri Değişlerime
################################################################################

int(b)
float(a)

a * b / 10
int(a * b / 10)

################################################################################
# Karakter Dizileri (Strings): str
################################################################################

print("John")
print('John')

"John"
name = "John"

################################################################################
# Çok Satırlı Karakter Dizileri
################################################################################

"""Veri Yapıları: Hızlı Özet,
Sayılar (Numbers): int, float, complex
Karakter Dizileri (Strings): str
List, Dictionary, Tuple, Set
Boolen (TURE-FALSE): bool"""

long_str = """Veri Yapıları: Hızlı Özet,
Sayılar (Numbers): int, float, complex
Karakter Dizileri (Strings): str
List, Dictionary, Tuple, Set
Boolen (TURE-FALSE): bool"""

################################################################################
# Karakter Dizilerinin Elemanlarına Erişmek
################################################################################

name = "John"
name[0]
name[3]
name[2]

################################################################################
# Karakter Dizilerinde Slice İşlemi
################################################################################

name[0:2]
name[1:]
name[:2]
name[-2:]
long_str[0:10]

################################################################################
# String İçerisinde Karakter Sorgulamak
################################################################################

"veri" in long_str # Case sensitive
"Veri" in long_str

"Atakan" in long_str

# "\n" aşağı satıra indirir

################################################################################
# String Metodları
################################################################################

dir(int)
dir(str)

len(name)
len(long_str)
type(len)
len("Atakan")

################################################################################
# upper() & lower()
################################################################################

"Atakan".upper()
"Atakan".lower()
name.upper()

################################################################################
# replace()
################################################################################

hi = "Hello AI Era"
hi.replace("l", "p")

################################################################################
# split()
################################################################################

"Hello AI Era".split()
"Hello AI Era".split("A")

################################################################################
# strip()
################################################################################

" ofofo ".split()
"ofofo".split("o")

################################################################################
# capitalize()
################################################################################

"atakan".capitalize()

dir(str)

################################################################################
# startswith()()
################################################################################

"atakan".startswith("A")
"atakan".startswith("a")

################################################################################
# Liste (List)
################################################################################

# Değiştirlebilir.
# Sıralıdır.
# Kapsayıcıdır.

notes = [1, 2, 3, 4]
type(notes)

names = ["a", "b", "v", "d"]
type(names)

not_nam = [1, 2, 3, "a", "b", True, [1, 2, 3]]
type(not_nam)

not_nam[0]

not_nam[5]

not_nam[6]
not_nam[6][1]

type(not_nam[6])
type(not_nam[6][1])

notes[0]
notes[0] = 99
notes

len(notes)
not_nam[0:4]

################################################################################
# List Methods
################################################################################

dir(list)
dir(notes)

################################################################################
# len()
################################################################################

len(notes)
len(not_nam)

################################################################################
# append() #sona ekler
################################################################################

notes
notes.append(100)
notes

################################################################################
# insert() #index, object
################################################################################

notes
notes.insert(1, 200)
notes

################################################################################
# pop() #index
################################################################################

notes
notes.pop(0)
notes

################################################################################
# remove() #object
################################################################################

notes
notes.remove(2)
notes

################################################################################
# reverse()
################################################################################

notes
notes.reverse()
notes

################################################################################
# sort()
################################################################################

notes
notes.sort()
notes

################################################################################
# Dictionary #key-value
################################################################################

# Değiştirilebilir.
# Artık Sıralı
# Kapsayıcıdır.

dic = {"REG": ["RMSE", 10],
       "LOG": "Logistic Regression",
       "CART": "Classification and Reg"}
type(dic)

dic["REG"]
dic["REG"][1]

################################################################################
# Key Sorgulama
################################################################################

"REG" in dic
"ATA" in dic

################################################################################
# Key'E Göre Value'a Erişmek
################################################################################

dic["REG"]
dic.get("REG")

################################################################################
# Value Değiştirmek
################################################################################

dic
dic["LOG"] = ["YSA", 33]
dic

################################################################################
# Tüm Key ve Value'lara Erişmek
################################################################################

dic.keys()
dic.values()
dic.items() # liste içerisinde tuple formatında tüm elemanlar

################################################################################
# Key-Value Değerlerini Değiştirmek ve Yenisini Eklemek
################################################################################

dir(dict)
dir(dic)
dic
dic.update({"LOG": 44})
dic

dic
dic.update({"ATA": "ATAKAN"})
dic


################################################################################
# Tuple
################################################################################

# Değiştirilemez.
# Sıralıdır.
# Kapsayıcıdır.

t = ("john", "mark", 1, 3)
type(t)

t[0]
t[0:3]

t[0] = 99

# değişiklik yapmanın yolu

t = list(t)
t[0] = 99
t = tuple(t)
type(t)
t

################################################################################
# Set
################################################################################

# Değiştirilebilir.
# Sırasız ve Eşsizdir.
# Kapsayıcıdır.

set1 = set([1, 3, 5])
set2 = {1, 2, 3}
type(set1)
type(set2)

set1
set2
dir(set)

################################################################################
# difference() # -
################################################################################

set1.difference(set2)  # set1 - set2
set2.difference(set1)

################################################################################
# symmetric_difference()
################################################################################

set1.symmetric_difference(set2)

################################################################################
# intersection()  # &
################################################################################

set1.intersection(set2)  # set1 & set2

################################################################################
# union()
################################################################################

set1.union(set2)

################################################################################
# isdisjoint()
################################################################################

set1.isdisjoint(set2)

################################################################################
# issubset()
################################################################################

set1.issubset(set2)
set2.issubset(set1)

################################################################################
# issuperset()
################################################################################

set1.issuperset(set2)
set2.issuperset(set1)

################################################################################
# FONKSİYONLAR, KOŞULLAR, DÖNGÜLER, COMPREHENSIONS
################################################################################

# - Fonksiyonlar (Functions)
# - Koşullar (Conditions)
# - Döngüler (Loops)
# - Comprehensions


################################################################################
# Functions
################################################################################

print("a")
?print
help(print)

print("atakan", "acıbal")

print("atakan", "acıbal", sep = "__")

################################################################################
# Fonksiyon Tanımlama
################################################################################

def calculate(x):
    print(x * 2)

calculate(5)

def summer(arg1, arg2):
    print(arg1 + arg2)

summer(4, 5)

def div(arg1, arg2):
    print(arg1 / arg2)

div(6, 2)
div(2, 6)
div(arg2=2, arg1=6)


################################################################################
# Docstring
################################################################################

def summer(arg1, arg2):
    print(arg1 + arg2)

def summer(arg1, arg2):
    """
    Sum of two numbers

    Parameters
    ----------
    arg1: int, float
    arg2: int, float

    Returns
    -------
    int, float
    """
    print(arg1 + arg2)

summer(4,8)
?summer
help(summer)

################################################################################
# Fonksiyonların Statement/Body Bölümü
################################################################################

# def funtion_name(parameters/agruments):
#       statements (function body)

def say_hi():
    print("Merhaba")
    print("Hi")
    print("hello")

say_hi()


def say_hi(string):
    print(string)
    print("Hi")
    print("hello")

say_hi()
say_hi("Slm")


def multiplication(a = 2, b = 3):
    c = a * b
    print(c)

multiplication()
multiplication(b = 5)
multiplication(a = 5)

multiplication(3,4)


list_store = []

def add_element(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)

list_store

add_element(3,5)

list_store

add_element(9,34)

list_store

# append atama işlemine gerek kalmaksızın kalıcı değişiklik yapar.


################################################################################
# Default Parameters/Arguments
################################################################################

def divide(a, b):
    print(a / b)

divide(1)
divide(1,2)


def divide(a = 1, b = 1):
    print(a / b)

divide(3)
divide(b = 3)
divide(b = 3, a = 9)

def say_hi(string = "Merhaba"):
    print(string)
    print("Hi")
    print("Hello")

say_hi()
say_hi("ATAKAN")

################################################################################
# Ne Zaman Fonksiyon Yazmaya İhtiyacımız Olur?
################################################################################

# warm, moisture, charge

(56 + 15) / 80
(17 + 45) / 60

# DRY (DON'T REPEAT YOURSELF)

def calculate(warm, moisture, charge):
    print((warm + moisture) / charge)

calculate(56, 15, 80)


################################################################################
# Return
################################################################################

def calculate(warm, moisture, charge):
    print((warm + moisture) / charge)

calculate(56, 15, 80)

calculate(56, 15, 80) * 10

type(calculate(56, 15, 80))

def calculate(warm, moisture, charge):
    return((warm + moisture) / charge)

def calculate(warm, moisture, charge):
    return (warm + moisture) / charge

# return'den sonraki kodlar çalışmaz. Return ile biter

type(calculate(56, 15, 80))

calculate(56, 15, 80)

calculate(56, 15, 80) * 10

def calculate(warm, moisture, charge):
    warm = warm * 2
    moisture = moisture * 2
    charge = charge * 2
    output = (warm + moisture) / charge

    return warm, moisture, charge, output

type(calculate(98, 12, 78))

calculate(98, 12, 78)

calculate(98, 12, 78) * 2 # tuple 2 ile çarpınca değerleri tekrar yazdırır.

warm, moisture, charge, output = calculate(98, 12, 78)
print(warm, moisture, charge, output)

################################################################################
# Fonksiyon İçerisinden Fonksiyon Çağırmak
################################################################################

def calculate(warm, moisture, charge):
    return ((warm + moisture) / charge)

calculate(90, 12, 12) * 10

def standardization(a, p):
    return a * 10 / 100 * p * p

standardization(45, 1)


def all_calculate(warm, moisture, charge, p):
    a = calculate(warm, moisture, charge)
    b = standardization(a, p)
    print(b * 10)

all_calculate(90, 12, 12, 2)


################################################################################
# Local & Global Variables
################################################################################

list_store = [1, 2]  # global

def add_element(a,b):
    c = a * b  # local
    list_store.append(c)
    print(list_store)

add_element(2, 3)

print(list_store)  # global
print(c)  # local

################################################################################
# KOŞULLAR (CONDITIONS)
################################################################################

# True-False

1 == 1
1 == 2
1 != 2

################################################################################
# if
################################################################################

if 1 == 1:
    print("something")


if 1 == 2:
    print("something")

number = 11

if number == 10:
    print("number is 10")


number = 10

if number == 10:
    print("number is 10")


def number_check(number):
    if number == 10:
        print("number is 10")

number_check(10)
number_check(15)

################################################################################
# else
################################################################################

def number_check(number):
    if number == 10:
        print("number is 10")
    else:
        print("number is not 10")

number_check(10)
number_check(15)

################################################################################
# elif
################################################################################

def number_check(number):
    if number > 10:
        print("number is greater than 10")
    elif number < 10:
        print("number is less than 10")
    else:
        print("number is equal to 10")

number_check(5)
number_check(10)
number_check(15)

################################################################################
# Döngüler (Loops)
################################################################################

# for loop

students = ["John", "Mark", "Venessa", "Mariam"]

students[0]
students[1]
students[2]
students[3]

# DRY (DON'T REPEAT YOURSELF)

for student in students:
    print(student)

for student in students:
    print(student.upper())

salaries = [1000, 2000, 3000, 4000, 5000]

for salary in salaries:
    print(salary)

for salary in salaries:
    print(int(salary*20/100 + salary))

def new_salary(salary, rate):
    return int(salary*rate/100 + salary)

new_salary(1000, 20)

for salary in salaries:
    print(new_salary(salary, 30))


for salary in salaries:
    if salary >= 3000:
        print(new_salary(salary, 10))
    else:
        print(new_salary(salary, 20))

################################################################################
# Uygulama
################################################################################

# before: "hi my name is john and i am learning python"
# after: "Hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

sentence = "hi my name is john and i am learning python"

def alternating(string):
    new_string = ""
    for index in range(len(string)):
        if index % 2 == 0:
            new_string += string[index].upper()
        else:
            new_string += string[index].lower()
    print(new_string)

alternating(sentence)

def alternating2(string):
    new_string = ""
    for index, letter in enumerate(string, 0):
        if index % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)

alternating2(sentence)


################################################################################
# Break, Continue, While
################################################################################

salaries = [1000, 2000, 3000, 4000, 5000]

# break

for salary in salaries:
    if salary == 3000:
        break
    print(salary)

# continue

for salary in salaries:
    if salary == 3000:
        continue
    print(salary)

# while

number = 1
while number < 5:
    print(number)
    number += 1

################################################################################
# Enumerate
################################################################################

students = ["John", "Mark", "Venessa", "Mariam"]

for index, student in enumerate(students):
    print(index, student)

for index, student in enumerate(students, 1):
    print(index, student)

A = []
B = []

for index, student in enumerate(students):
    if index % 2 == 0:
        A.append(student)
    else:
        B.append(student)
    print(A, B)

A
B


students = ["John", "Mark", "Venessa", "Mariam"]

def divide_students(students):
    groups = [[], []]
    for index, student in enumerate(students):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    return groups

st = divide_students(students)
st


sentence = "hi my name is john and i am learning python"

def alternating2(string):
    new_string = ""
    for index, letter in enumerate(string, 0):
        if index % 2 == 0:
            new_string += letter.upper()
        else:
            new_string += letter.lower()
    print(new_string)

alternating2(sentence)

################################################################################
# Zip
################################################################################

students = ["John", "Mark", "Venessa", "Mariam"]

departments = ["mathmatics", "statistics", "physics", "astronomy"]

ages = [23, 30, 26, 22]

all_list = list(zip(students, departments, ages))
print(all_list)
#liste içerisinde tuple formunda

################################################################################
# lambda, map, filter, reduce
################################################################################

# lambda

def summer(a, b):
    return a + b

summer(1, 3) * 9

new_sum = lambda a, b: a + b

new_sum(1,3)

# map

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

for salary in salaries:
    print(new_salary(salary))


list(map(new_salary, salaries))


list(map(lambda x: x * 30 / 100 + x, salaries))

list(map(lambda x: x ** 2, salaries))

# filter

list_store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

list(filter(lambda x: x % 2 == 0, list_store))

list(filter(lambda x: x > 5, list_store))
# atama yok

# reduce

from functools import reduce

reduce(lambda a, b: a + b, list_store)

################################################################################
# COMPREHENSIONS
################################################################################

################################################################################
# List Comprehension
################################################################################

salaries = [1000, 2000, 3000, 4000, 5000]

def new_salary(x):
    return x * 20 / 100 + x

for salary in salaries:
    print(new_salary(salary))

null_list = []

for salary in salaries:
    null_list.append(new_salary(salary))

null_list = []

for salary in salaries:
    if salary > 3000:
        null_list.append(new_salary(salary))
    else:
        null_list.append(new_salary(salary * 2))


[new_salary(salary * 2) if salary < 3000 else new_salary(salary) for salary in salaries]

[salary * 2 for salary in salaries]

[salary * 2 for salary in salaries if salary < 3000]

[salary * 2 if salary < 3000 else salary * 0 for salary in salaries]

[new_salary(salary * 2) if salary < 3000 else new_salary(salary * 0.2) for salary in salaries]

students = ["John", "Mark", "Venessa", "Mariam"]

students_no = ["John", "Venessa"]

[student.upper() if student not in students_no else student.lower() for student in students]

################################################################################
# Dick Comprehension
################################################################################

dic = {"a": 1,
       "b": 2,
       "c": 3,
       "d": 4}

dic.keys()
dic.values()
dic.items()

{k: v ** 2 for (k, v) in dic.items()}

{k.upper(): v ** 2 for (k, v) in dic.items()}


################################################################################
# Uygulama
################################################################################

numbers = range(10)

new_dic = {}

for number in numbers:
    if number % 2 == 0:
        new_dic[number] = number ** 2

{number: number ** 2 for number in numbers if number % 2 == 0}

################################################################################
# Uygulama
################################################################################

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns
df.head()

for col in df.columns:
    print(col.upper())

A = []
for col in df.columns:
    A.append(col.upper())

df.columns = A

df.columns = [col.upper() for col in df.columns]
df.head()

################################################################################
# Uygulama
################################################################################

df = sns.load_dataset("car_crashes")
df.columns
df.head()

df.columns = [col.upper() for col in df.columns]
df.columns = ["FLAG_" + col if "INS" in col else "NO_FLAG_" + col for col in df.columns]

################################################################################
# Uygulama
################################################################################

df = sns.load_dataset("car_crashes")
df.columns
df.head()

num_cols = [col for col in df.columns if df[col].dtype != "O"]
dic = {}
agg_list = ["mean", "min", "max", "var"]

for col in num_cols:
    dic[col] = agg_list

dic = {col: agg_list for col in num_cols}

df.head()

df[num_cols].head()

df[num_cols].agg(dic)



################################################################################
# PYTHON İLE VERİ ANALİZİ (DATA ANALYSIS WITH PYTHON)
################################################################################

# - NumPy
# - Pandas
# - Veri Görselleştirme: Matplotlib & Seaborn
# - Advanced Functional Exploratory Data Analysis

################################################################################
# NUMPY
################################################################################

# Why NumPy?
# Creating NumPy Arrays
# Attributes of NumPy Arrays
# Reshaping
# Index Selection
# Slicing
# Fancy Index
# Conditions on NumPy
# Mathematical Operations

# Why NumPy?

import numpy as np
a = [1, 2, 3, 4]
b = [2, 3, 4, 5]

ab = []

for i in range(len(a)):
    ab.append(a[i] * b[i])

print(ab)

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a * b

################################################################################
# Creating NumPy Arrays
################################################################################

np.array([1, 2, 3, 4, 5])
type(np.array([1, 2, 3, 4, 5]))

np.array([[1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5]]) # manuel 1'den fazla satırlı array yapmak için çift [[..]] kullanılıyor.

np.zeros(shape=10, dtype=int)
np.zeros(shape=(5,4), dtype=int)

np.ones(shape=10, dtype=int)
np.ones(shape=(5,4), dtype=int)

np.random.randint(0, 10, size=10)
np.random.randint(2, 5, size=(3,4))

np.random.normal(10, 4, size=10)
np.random.normal(10,4, size=(3,4))

np.arange(0, 30, 3)
np.arange(0, 30, 2)
np.arange(0, 30, 2).reshape(3,5)


################################################################################
# Attributes of NumPy Arrays
################################################################################

# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10, size=5)

a.ndim
a.shape
a.size
a.dtype

################################################################################
# Reshaping
################################################################################

np.zeros(shape=10, dtype=int).reshape(2,5)
np.random.randint(0, 10, size=20).reshape(4,5)
np.random.normal(10, 4, size=6).reshape(2,3)

np.random.normal(10,4, size=(3,4)).reshape(4,3)
# boyutun uyması lazım matris yaparken

ar = np.random.randint(1, 10, size=9)
print(ar)
ar.reshape(3,3)

################################################################################
# Index Selection
################################################################################

a = np.random.randint(10, size=10)

a[0]
a[:5]

m = np.random.randint(10, size=(3,5))

m[0,0]
m[2,4]

m[1,1] = 999
m[1,1] += 1


m[1,1] = 31.124 #fix type
print(m)

m[:,0]
m[1,:]
m[1:3,2:]

################################################################################
# Fancy Index
################################################################################

v = np.arange(0,30,3)
v[1]
v[4]

catch = [2, 4, 7]
v[catch]

################################################################################
# Conditions on NumPy
################################################################################

v = np.array([1, 2, 3, 4, 5])

ab = []

for i in v:
   if i < 3:
       ab.append(i)

print(ab)

v > 3

v[v<3]

v[v==3]

v[v>=3]

################################################################################
# Mathematical Operations
################################################################################

v = np.array([1, 2, 3, 4, 5])

v / 5

v * 5 / 10

v ** 2

v + 2

v - 3

np.subtract(v, 1)
dir(v)
v.sum()
v.cumsum()

np.add(v, 2)

np.mean(v)
v.mean()

v.min()
np.min(v)

v.max()
np.max(v)

v.std()
np.std(v)

v.var()
np.var(v)

for i in v:
    print(math.sqrt(i))

# Denklem Çözümü

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5,1],[1,3]])
b = np.array([12,10])

np.linalg.solve(a,b)

# 5*1.85714286 + 1*2.71428571 = 12
# 1*1.85714286 + 3*2.71428571 = 10


################################################################################
# PANDAS
################################################################################

# Pandas Series
# Reading Data
# Quick Look at Data
# Selection in Pandas
# Aggregation & Grouping
# Apply & Lambda
# Join

################################################################################
# Pandas Series
################################################################################

import pandas as pd

s = pd.Series([10, 77, 12, 4, 5, 5])

print(s)
type(s)
s.index
dir(s)
s.dtype
s.size
s.ndim
s.values
s.unique()
type(s.values)
type(s.unique())

# values ve unique değerlerini çağırdığımızda numpy array'i olarak gelir.

s.head(3)
s.tail(3)
s.sum()

################################################################################
# Reading Data
################################################################################

import pandas as pd
df = pd.read_csv("datasets/advertising.csv")
df.head()

df = pd.read_excel("datasets/online_retail_II.xlsx")
df.head()

################################################################################
# Quick Look at Data
################################################################################

import seaborn as sns

df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df["sex"].value_counts()
df.sex.value_counts()

################################################################################
# Selection in Pandas
################################################################################

import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13]
df.drop(0, axis=0)
df.head()

delete_indexes = [1, 3, 5, 7]
df.drop(delete_indexes, axis=0).head(10)
df.head()

df = df.drop(delete_indexes, axis=0)
df.drop(delete_indexes, axis=0, inplace=True)

################################################################################
# Index'i Değişkene, Değişkeni Index'e Çevirmek
################################################################################

import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")
df.head()

df.reset_index(inplace=True)
df.index = df["age"]
df.drop("age", axis=1, inplace=True)
df.columns

df.reset_index(inplace=True)
df.head()
df["age2"] = df["age"] * 2
df[["age", "age2"]]

################################################################################
# Değişkenler Üzerinde İşlemler
################################################################################

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
df = sns.load_dataset("titanic")
df.head()

"age" in df

df["age"]
df.age

type(df["age"])  # pandas.Series

type(df[["age"]])  # pandas.DataFrame

df[["age", "alive"]] # pandas.DataFrame

col_names = ["age", "adult_male", "alive"]
df[col_names] # pandas.DataFrame

df["Atakan"] = 18
df.head()
df["Acibal"] = df["age"] ** 2
df.head()
df.drop("Acibal", axis=1, inplace=True)

df.drop(col_names, axis=1)

[df.drop(col, axis=1) for col in df.columns if "age" in col]
df.head()

df.loc[:, ~df.columns.str.contains("sex")]

################################################################################
# iloc & loc
################################################################################

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df[0:3]
df.iloc[0:3]
df.iloc[0,0]
df.iloc[1,2]
df.head()

df.loc[0:3] # loc'da 3'e kadar değil 3 dahildir.
df.loc[2:15, "age"]

df.iloc[0:4,0:3]
# or
col_names = ["survived", "pclass", "sex"]
df.loc[0:3, col_names]

################################################################################
# Conditional Selection
################################################################################

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df[df["age"] > 50]
df[df["age"] > 50].count()
df[df["age"] > 50]["age"].count()

df.loc[df["age"] > 50, "age"]
df.loc[df["age"] > 50, ["age", "pclass"]]
df.loc[(df["age"] > 50) & (df["sex"] == "male"), ["age", "pclass", "sex"]]
df.loc[(df["age"] > 50) & (df["sex"] == "male") & ((df["embark_town"] == "Southampton") | (df["embark_town"] == "Cherbourg")), ["age", "embark_town", "pclass", "sex"]]
df_new = df.loc[(df["age"] > 50) & (df["sex"] == "male") & ((df["embark_town"] == "Southampton") | (df["embark_town"] == "Cherbourg")), ["age", "embark_town", "pclass", "sex"]]
df_new.head()
df_new["embark_town"].value_counts()

################################################################################
# Aggregation & Grouping
################################################################################

# count()
# first()
# last()
# mean()
# median()
# min()
# max()
# std()
# var()
# pivot table

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()
df.groupby("sex")["age"].mean()
df.groupby("sex").agg({"age":["mean", "sum"]})
df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": ["mean", "sum"]})

df.groupby(["sex", "embark_town"]).agg({"age": "mean",
                                   "survived": ["mean", "sum"]})


df.groupby(["sex", "embark_town", "pclass"]).agg({"age": "mean",
                                   "survived": ["mean", "sum"],
                                   "sex": "count"})

################################################################################
# Pivot Table
################################################################################

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df.pivot_table("survived", "sex", "embarked")
# default değer ortalamadır. mean

df.pivot_table("survived", "sex", "embarked", aggfunc="std")

df.pivot_table("survived", "sex", ["embarked", "class"])

df.pivot_table("survived", ["sex","embark_town"], ["embarked", "class"])

df["new_age"] = pd.cut(df["age"], [0,10,18,25,40,90])
df.head()
df.new_age.value_counts()

df.pivot_table("survived", "sex", "new_age")

df.pivot_table("survived", "sex", ["new_age", "class"])

################################################################################
# Apply & Lambda
################################################################################

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5
df.head()

df["age"]/10
df["age2"]/10

[df[col]/10 for col in df.columns if "age" in col]
df.head()

df.loc[:,df.columns.str.contains("age")].apply(lambda x: x / 20)
df.head()

df.loc[:,df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std())

def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)
df.loc[:, df.columns.str.contains("age")] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler)
# df["age"] = df.loc[:, df.columns.str.contains("age")].apply(standart_scaler) # tek age varsa bu, yoksa üstteki
df.head()

################################################################################
# Join
################################################################################

import numpy as np
import pandas as pd
m = np.random.randint(1, 30, size=(5,3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2], axis=0)
pd.concat([df1, df2], axis=0, ignore_index=True)

pd.concat([df1, df2], axis=1) # saçma


df1 = pd.DataFrame({"employees":["john", "dennis", "mark", "maria"],
                   "group": ["accounting", "engineering", "engineering", "hr"]})

df2 = pd.DataFrame({"employees": ["mark", "john", "dennis", "maria"],
                    "start_date": [2010, 2009, 2014, 2019]})

df3 = pd.merge(df1, df2, on="employees")

df4 = pd.DataFrame({"group": ["accounting", "engineering", "hr"],
                   "manager": ["Caner", "Mustafa", "Berkcan"]})

df5 = pd.merge(df3, df4)
df5.head()


################################################################################
# VERİ GÖRSELLEŞTİRME: Matplotlib & Seaborn
################################################################################

################################################################################
# Matplotlib
################################################################################

# Kategorik: Sütun Grafik (bar & countplot)
# Sayısal: (hist & boxplot)

################################################################################
# Kategorik Değişken Görselleştirme
################################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind="bar")
plt.show()

plt.bar(x = df["sex"].value_counts().index, height = df["sex"].value_counts())
plt.show()

################################################################################
# Sayısal Değişken Görselleştirme
################################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()


df["age"].plot(kind="hist")
plt.show()

plt.hist(df["age"])
plt.show()

df["fare"].plot(kind="box")
plt.show()

plt.boxplot(df["fare"])
plt.show()

################################################################################
# Matplotlib Özellikleri
################################################################################

# katmanlıdır.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

################################################################################
# plot
################################################################################

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, "o") # böyle sadece nokta, marker="o" yapınca hem nokta hem çizgi
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])

plt.plot(x, y)
plt.show()

x2 = [1, 8]
y2 = [0, 150]

plt.plot(x2, y2)
plt.show()

plt.plot(x, y, "o")
plt.show()

################################################################################
# marker
################################################################################

y = np.array([13, 28, 11, 100])

plt.plot(y, marker="o")
plt.show()

plt.plot(y, marker="*")
plt.show()

markers = ["o", "*", ".", ",", "x", "X", "+", "P", "s", "D", "d", "p", "H", "h"]

################################################################################
# line
################################################################################

y = np.array([13, 28, 11, 100])

plt.plot(y)
plt.show()

plt.plot(y, ls="dashdot")
plt.show()

ls = ["dotted", "dashed", "dashdot"]

plt.plot(y, ls="dashdot", color="g")
plt.show()

color = ["b", "g", "r", "c", "m", "k", "w"]

################################################################################
# multible lines
################################################################################

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()

################################################################################
# labels
################################################################################

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x,y)
plt.title("bu bir başlıktır.")
plt.xlabel("X ekseni")
plt.ylabel("Y ekseni")
plt.grid()
plt.show()

# Jupyter'de sonuna ; işareti konulur çıktı düzgün çıksın diye

################################################################################
# subplots
################################################################################

x = np.array([80, 85, 90,95, 340, 105, 120, 115, 120, 125])
y = np.array([240, 287, 260, 270, 580, 290, 700, 310, 320, 330])
plt.subplot(1, 2, 1)
plt.title("1")
plt.plot(x, y)


x = np.array([80, 85, 90,95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(x, y)

################################################################################
# Seaborn
################################################################################

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
df = sns.load_dataset("tips")
df.head()

################################################################################
# Kategorik Değişken Görselleştirme
################################################################################

df["sex"].value_counts()
sns.countplot(x="sex", data=df)
#or
sns.countplot(x=df["sex"])

# matplotlib
df["sex"].value_counts().plot(kind="bar")
plt.show()
#or
plt.bar(x = df["sex"].value_counts().index, height = df["sex"].value_counts())
plt.show()

################################################################################
# Sayısal Değişken Görselleştirme
################################################################################

sns.boxplot(x="total_bill", data=df)  # yatay
#or
sns.boxplot(x=df["total_bill"])

# matplotlib
plt.boxplot(df["total_bill"])  # dik
plt.show()
#or
df["total_bill"].plot(kind="box")
plt.show()


sns.histplot(x="total_bill", data=df)
#or
sns.histplot(x=df["total_bill"])

# matplotlib
plt.hist(df["total_bill"])
plt.show()
#or
df["total_bill"].plot(kind="hist")
plt.show()

# pandas function
df["total_bill"].hist()

################################################################################
# Advanced Functional Exploratory Data Analysis
################################################################################

# Big Picture
# Analysis of Categorical Variables
# Analysis of Numerical Variables
# Analysis of Target Variable
# Analysis of Correlation

################################################################################
# Big Picture
################################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.dtypes
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

def check_df(dataframe, head=5, tail=5):
    print("###################### Shape ######################")
    print(dataframe.shape)
    print("###################### Types ######################")
    print(dataframe.dtypes)
    print("###################### Head ######################")
    print(dataframe.head(head))
    print("###################### Tail ######################")
    print(dataframe.tail(tail))
    print("#################### NA ####################")
    print(dataframe.isnull().sum())
    print("#################### Quantiles ####################")
    print(dataframe.describe([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).T)


check_df(df)

df2 = sns.load_dataset("tips")
check_df(df2)

df3 = sns.load_dataset("flights")
check_df(df3)

################################################################################
# Analysis of Categorical Variables
################################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["embarked"].value_counts()
df["sex"].unique()
df["class"].nunique()

df.info()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

num_but_cat = [col for col in df.columns if (df[col].dtypes in ["int", "float"]) & (df[col].nunique() < 10)]

cat_but_car = [col for col in df.columns if (str(df[col].dtypes) in ["category", "object"]) & (df[col].nunique() > 20)]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("################################################")

cat_summary(dataframe=df, col_name="sex")


for col in cat_cols:
    cat_summary(df, col)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(dataframe=df, col_name="sex", plot=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)


for col in cat_cols:
    if df[col].dtypes == "bool":
        print("Type is bool")
    else:
        cat_summary(df, col, plot=True)

df["adult_male"].astype(int)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)

################################################################################
# Analysis of Numerical Variables
################################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df[["age", "fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]


cat_cols = [col for col in df.columns if str(df[col].dtypes) in  ["category", "object", "bool"]]

num_but_cat = [col for col in df.columns if (df[col].dtypes in ["int", "float"]) & (df[col].nunique() < 10)]

cat_but_car = [col for col in df.columns if (str(df[col].dtypes) in ["category", "object"]) & (df[col].nunique() > 5)]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]


num_cols = [col for col in num_cols if col not in cat_cols]

def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        dataframe[col_name].hist()
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show(block=True)

num_summary(dataframe=df, col_name="age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)

################################################################################
# Capturing Variables and Generalizing Operations
################################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Değişkenleri ayırır.
    Parameters
    ----------
    dataframe : dataframe
        Veriseti
    cat_th : int, float
        Kategorik Değişken Tresshold
    car_th : int, float
        Kardinal Değişken Tresshold

    Returns
        cat_cols
        num_cols
        cat_but_car

    Notes
        cat_cols + num_cols + car_but_car = toplam değişken sayısı
    -------

    """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if (dataframe[col].dtypes in ["int", "float"]) & (dataframe[col].nunique() < cat_th)]
    cat_but_car = [col for col in dataframe.columns if (str(dataframe[col].dtypes) in ["category", "object"]) & (dataframe[col].nunique() > car_th)]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables {dataframe.shape[1]}")
    print(f"cat_cols {len(cat_cols)}")
    print(f"num_cols {len(num_cols)}")
    print(f"cat_but_car {len(cat_but_car)}")
    print(f"num_but_cat {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df,10,20)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)


def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    print(dataframe[col_name].describe(quantiles).T)

    if plot:
        dataframe[col_name].hist()
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)

################################################################################
# Analysis of Target Variable
################################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

# ya başında bunu yapıp bool'lardan kurtulacan, ya da cat_summary de düzeltecen
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Değişkenleri ayırır.
    Parameters
    ----------
    dataframe : dataframe
        Veriseti
    cat_th : int, float
        Kategorik Değişken Tresshold
    car_th : int, float
        Kardinal Değişken Tresshold

    Returns
        cat_cols
        num_cols
        cat_but_car

    Notes
        cat_cols + num_cols + car_but_car = toplam değişken sayısı
    -------

    """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if (df[col].dtypes in ["int", "float"]) & (df[col].nunique() < cat_th)]
    cat_but_car = [col for col in df.columns if (str(df[col].dtypes) in ["category", "object"]) & (df[col].nunique() > car_th)]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables {dataframe.shape[1]}")
    print(f"cat_cols {len(cat_cols)}")
    print(f"num_cols {len(num_cols)}")
    print(f"cat_but_car {len(cat_but_car)}")
    print(f"num_but_cat {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df,10,20)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("################################################")

    if plot:
        sns.countplot(x=dataframe[col_name])
        plt.show(block=True)


for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)


def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    print(dataframe[col_name].describe().T)

    if plot:
        dataframe[col_name].hist()
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show(block=True)


for col in num_cols:
    num_summary(df, col, plot=True)


df.head()
df["survived"].value_counts()

cat_summary(dataframe=df, col_name="survived")

################################################################################
# Hedef Değişkenin Kategorik Değişkenler İle Analizi
################################################################################

df.groupby("sex")["survived"].mean()

def target_summary_with_cat(dataframe, target, col_name):
    print(dataframe.groupby(col_name).agg({target:"mean"}))

target_summary_with_cat(df, "survived", "sex")

target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)

################################################################################
# Hedef Değişkenin Sayısal Değişkenler İle Analizi
################################################################################

df.groupby("survived")["age"].mean()

def target_summary_with_num(dataframe, target, col_name):
    print(dataframe.groupby(target).agg({col_name:"mean"}))

target_summary_with_num(df, "survived", "age")

target_summary_with_num(df, "survived", "fare")

for col in num_cols:
    target_summary_with_num(df, "survived", col)

################################################################################
# Analysis of Correlation
################################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("DS_Bootcamp_13/datasets/breast_cancer.csv")
df = df.iloc[:,1:-1] # or df.drop("id", axis=1, inplace=True)
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

corr = df[num_cols].corr()

sns.set(rc={"figure.figsize": (10, 10)})
sns.heatmap(corr, cmap="RdBu")
plt.show

################################################################################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi
################################################################################

corr_matrix = df[num_cols].corr().abs()

upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]


def high_correlated_cols(dataframe, num_cols, plot=False, corr_th=0.90):
    corr = dataframe[num_cols].corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize":  (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

drop_list = high_correlated_cols(df, num_cols)

high_correlated_cols(df, plot=True)

len(df.drop(drop_list, axis=1).columns) # drop'tan sonra kalan kolonların sayısı

len(drop_list) # silinen kolonların sayısı

################################################################################
# Introduction to CRM Analytics
################################################################################

# CRM: Customer Relationship Management
# Customer Lifecycle/Journey/Funnel
# KPIs: Key Performance Indicators
    # Customer Acquisition Rate
    # Customer Retention Rate
    # Customer Churn Rate
    # Conversion Rate
    # Growth Rate
# Analysis of Cohort


################################################################################
# Customer Segmentation with RFM
################################################################################

# RFM Metrics: Recency, Frequency, Monetary

###############################################################
# 1. Business Problem
###############################################################

# Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejileri belirlemek istiyor.

# Veri Seti Hikayesi
# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Online Retail II isimli veri seti İngiltere merkezli online bir satış mağazasının
# 01/12/2009 - 09/12/2011 tarihleri arasındaki satışlarını içeriyor.

# Değişkenler

# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

################################################################################
# Data Understanding
################################################################################

import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 200)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_ = pd.read_excel("DS_Bootcamp_13/datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.shape
df.isnull().sum()

# essiz urun sayisi nedir?
df["Description"].nunique() #4681
df["StockCode"].nunique() # 4632
df["Description"].value_counts().head()

df.groupby("Description").agg({"Quantity": "sum"}).head()

df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

df["Invoice"].nunique()

df["TotalPrice"] = df["Quantity"] * df["Price"]

df.groupby("Invoice").agg({"TotalPrice": "sum"}).head()

df.groupby("Invoice").agg({"TotalPrice": "sum"}).sort_values("TotalPrice", ascending=False).head()

###############################################################
# 3. Data Preparation
###############################################################

df.shape
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()

df.describe().T

df = df[~df["Invoice"].str.contains("C", na=False)]
df.describe().T
# "C" olanlar iadeler gitti ama hala "Price"'ı 0 olanlar var onları ya burda ya da ilerde yok edeceğiz.

###############################################################
# 4. Calculating RFM Metrics
###############################################################

# Recency, Frequency, Monetary
df.head()
df["InvoiceDate"].max()
today_date = dt.datetime(2010, 12, 11)
type(today_date)

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda x: (today_date - x.max()).days,
                                     'Invoice': lambda x: x.nunique(),
                                     'TotalPrice': lambda x: x.sum()})

rfm.head()
rfm.columns = ["recency", "frequency", "monetary"]
rfm.head()

rfm.describe().T

rfm = rfm[rfm["monetary"] > 0]

rfm.describe().T

###############################################################
# 5. Calculating RFM Scores
###############################################################

rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])

rfm.head()

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm.head()

rfm["monatery_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm.head()

rfm["RFM_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

rfm.head()

rfm.describe().T

rfm[rfm["RFM_SCORE"] == "55"]

rfm[rfm["RFM_SCORE"] == "11"]

###############################################################
# 6. Creating & Analysing RFM Segments
###############################################################
# regex

# RFM segments
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex=True)
rfm.head()

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg({"recency": ["mean", "count"],
                                                                             "frequency": ["mean", "count"],
                                                                             "monetary": ["mean", "count"]})

rfm[rfm["segment"] == "cant_loose"]
rfm[rfm["segment"] == "cant_loose"].index

new_df = pd.DataFrame()

new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index

new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)

new_df.to_csv("new_customers.csv")

rfm.to_csv("rfm.csv")

###############################################################
# 7. Tüm Sürecin Fonksiyonlaştırılması
###############################################################

def create_rfm(dataframe, csv=False):

    # VERIYI HAZIRLAMA
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # RFM METRIKLERININ HESAPLANMASI
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # cltv_df skorları kategorik değere dönüştürülüp df'e eklendi
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))


    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    rfm.index = rfm.index.astype(int)

    if csv:
        rfm.to_csv("rfm.csv")

    return rfm

df = df_.copy()

rfm_new = create_rfm(df)


############################################
# CUSTOMER LIFETIME VALUE
############################################

# 1. Data Preparation
# 2. Average Order Value (average_order_value = total_price / total_transaction)
# 3. Purchase Frequency (total_transaction / total_number_of_customers)
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
# 5. Profit Margin (profit_margin =  total_price * 0.10)
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
# 8. Creating Segments
# 9. BONUS: Tüm İşlemlerin Fonksiyonlaştırılması

##################################################
# 1. Data Preparation
##################################################

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)
pd.set_option("display.width", 300)

df_ = pd.read_excel("DS_Bootcamp_13/datasets/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()

df.isnull().sum()

df = df[~df["Invoice"].str.contains("C", na=False)]

df.describe().T

df = df[df["Quantity"] > 0]

df.describe().T

df.dropna(inplace=True)

df.describe().T

df["TotalPrice"] = df["Price"] * df["Quantity"]

df.head()

cltv_c = df.groupby("Customer ID").agg({"Invoice": lambda x: x.nunique(),
                                        "Quantity": lambda x: x.sum(), #gözlemlemek için
                                        "TotalPrice": lambda x: x.sum()})

cltv_c.columns = ["total_transaction", "total_unit", "total_price"]

cltv_c.head()

##################################################
# 2. Average Order Value (average_order_value = total_price / total_transaction)
##################################################

cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

cltv_c.head()

##################################################
# 3. Purchase Frequency (purchase_frequency = total_transaction / total_number_of_customers)
##################################################

cltv_c.shape[0]

cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

cltv_c.head()

##################################################
# 4. Repeat Rate & Churn Rate (repeat_rate = birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler)
##################################################

repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c[cltv_c["total_transaction"] > 0].shape[0]

churn_rate = 1 - repeat_rate

##################################################
# 5. Profit Margin (profit_margin =  total_price * 0.10)
##################################################

cltv_c["profit_margin"] = cltv_c["total_price"] * 0.10

cltv_c.head()

##################################################
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)
##################################################

cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]
# "customer value" değerini kısa yoldan cltv_c["customer_value_2"] = cltv_c["total_price"] / cltv_c.shape[0] ile de bulabilirdik. Ama "total_transaction" repeat_rate için zaten lazım olacaktı.
cltv_c.head()

##################################################
# 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
##################################################

cltv_c["CLTV"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

cltv_c.head()

cltv_c.sort_values(by="CLTV", ascending=False)

cltv_c["total_price"].max()

cltv_c.describe().T

##################################################
# 8. Creating Segments
##################################################

cltv_c.head()

cltv_c.sort_values(by="CLTV", ascending=False)

cltv_c["segment"] = pd.qcut(cltv_c["CLTV"], 4, labels= ["D", "C", "B", "A"])

cltv_c.head()

cltv_c.sort_values(by="CLTV", ascending=False)

cltv_c.groupby("segment").agg({"count", "mean", "sum"})

cltv_c.to_csv("cltv_c.csv")

##################################################
# 9. BONUS: Tüm İşlemlerin Fonksiyonlaştırılması
##################################################

def create_cltv_c(dataframe, profit=0.10, csv=False):

    # Veriyi hazırlama
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[(dataframe['Quantity'] > 0)]
    dataframe.dropna(inplace=True)
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    cltv_c = dataframe.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                                   'Quantity': lambda x: x.sum(),
                                                   'TotalPrice': lambda x: x.sum()})
    cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']
    # avg_order_value
    cltv_c['avg_order_value'] = cltv_c['total_price'] / cltv_c['total_transaction']
    # purchase_frequency
    cltv_c["purchase_frequency"] = cltv_c['total_transaction'] / cltv_c.shape[0]
    # repeat rate & churn rate
    repeat_rate = cltv_c[cltv_c.total_transaction > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate
    # profit_margin
    cltv_c['profit_margin'] = cltv_c['total_price'] * profit
    # Customer Value
    cltv_c['customer_value'] = (cltv_c['avg_order_value'] * cltv_c["purchase_frequency"])
    # Customer Lifetime Value
    cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']
    # Segment
    cltv_c["segment"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])

    if csv:
        cltv_c.to_csv("cltv_c.csv")

    return cltv_c


df = df_.copy()

cltv_c_new = create_cltv_c(df)

##############################################################
# CLTV PREDICTION
##############################################################

# CLTV = Conditional Expected Number of Transaction * Conditional Expected Average Profit
# CLTV = BG/NBD Model * Gamma Gamma Submodel

##############################################################
# Conditional Expected Number of Transaction with BG/NBD Model (Beta Geometric / Negative Binomial Distribution)
##############################################################

# BG/NBD = Buy Till You Die
# Transaction Process (Buy) + Dropout Process (Till you die)

# Transaction Process (Buy): Poisson Distribution (1 customer), Gamma Distribution (all customers)
# Dropout Process (Till you die): Beta Distribution (all customers)

##############################################################
# Conditional Expected Average Profit with Gamma Gamma Submodel
##############################################################

##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

# 1. Data Preperation
# 2. BG-NBD Modeli ile Expected Number of Transaction
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 6. Çalışmanın fonksiyonlaştırılması

##############################################################
# 1. Data Preparation
##############################################################

# Gerekli Kütüphane ve Fonksiyonlar

# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda x: "%.4f" % x)


def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.05)
    quartile3 = dataframe[col_name].quantile(0.95)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

#########################
# Data Reading
#########################

df_ = pd.read_excel("DS_Bootcamp_13/datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.describe().T

#########################
# Data Preprocessing
#########################

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df.head()
df.describe().T

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]

df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

#########################
# Lifetime Veri Yapısının Hazırlanması
#########################

# recency : Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency)
# monetary: satın alma başına ortalama kazanç

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [lambda x: (x.max() - x.min()).days,
                                                         lambda x: (today_date - x.min()).days],
                                         "Invoice": lambda x: x.nunique(),
                                         "TotalPrice": lambda x: x.sum()})

cltv_df.head()
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]


cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df[(cltv_df["frequency"] > 1)]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

cltv_df.describe().T

##############################################################
# 2. BG-NBD Modelling
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])


################################################################
# 1 hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending=False)


cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df["frequency"],
                                              cltv_df["recency"],
                                              cltv_df["T"]).sort_values(ascending=False)

cltv_df.head()

################################################################
# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                              cltv_df["frequency"],
                                              cltv_df["recency"],
                                              cltv_df["T"]).sort_values(ascending=False)

cltv_df.head()

################################################################
# 3 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################

bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

################################################################
# Tahmin Sonuçlarının Değerlendirilmesi
################################################################

plot_period_transactions(bgf)
plt.show()

##############################################################
# 3. GAMMA-GAMMA Modelling
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"],
        cltv_df["monetary"])

ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).sort_values(ascending=False)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                             cltv_df["monetary"])

cltv_df.sort_values("expected_average_profit", ascending=False)

##############################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################

cltv = ggf.customer_lifetime_value(bgf,
                                    cltv_df["frequency"],
                                    cltv_df["recency"],
                                    cltv_df["T"],
                                    cltv_df["monetary"],
                                    time=3, #3 aylık
                                    freq="W", #T'nin frekans bilgisi
                                    discount_rate=0.01)

cltv.head()

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")

cltv_final.head()

cltv_final.sort_values("clv", ascending=False) # bu 3 aylık clv değerleri

##############################################################
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
##############################################################

cltv_final.head(50)

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(50)

cltv_final.groupby("segment").agg({"count", "mean", "sum"})


##############################################################
# 6. Çalışmanın Fonksiyonlaştırılması
##############################################################

def create_cltv_p(dataframe, month=3, csv=False):
    # 1. Veri Ön İşleme
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(1)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    # 3. GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    if csv:
        cltv_final.to_csv("cltv_final.csv")

    return cltv_final


df = df_.copy()

cltv_final2 = create_cltv_p(df)

cltv_final2.to_csv("cltv_prediction.csv")


###################################################
# Rating Products
###################################################

# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating


############################################
# Uygulama: Kullanıcı ve Zaman Ağırlıklı Kurs Puanı Hesaplama
############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 20)
pd.set_option("display.width", 300)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# (50+ Saat) Python A-Z™: Veri Bilimi ve Machine Learning
# Puan: 4.8 (4.764925)
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6

df = pd.read_csv("DS_Bootcamp_13/datasets/course_reviews.csv")
df.head()
df.shape

df["Rating"].value_counts()

df["Questions Asked"].value_counts()

df[df["Questions Asked"] == 22]["Rating"]

df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})

df.head()

####################
# Average
####################

df["Rating"].mean() #4.7642

####################
# Time-Based Weighted Average
####################

df.head()
df.info()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df.info()

df["Timestamp"].max()

current_date = pd.to_datetime("2021-02-10 00:00:00")

df["days"] = (current_date - df["Timestamp"]).dt.days

df.head()
df.tail()

df[df["days"] <= 30]["Rating"].mean() #4.7757

df.loc[df["days"] <= 30, "Rating"].mean()
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()
df.loc[(df["days"] > 180), "Rating"].mean()



df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
df.loc[(df["days"] > 180), "Rating"].mean() * 22/100

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "Rating"].mean() * w4 / 100


time_based_weighted_average(df) #4.7650
time_based_weighted_average(df, 30, 26, 22, 22)

####################
# User-Based Weighted Average
####################

df.head()

df["Progress"].head()

df.groupby("Progress").agg({"Progress": "count",
                           "Rating": "mean"})


df.loc[df["Progress"] <= 10, "Rating"].mean()
df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean()
df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean()
df.loc[df["Progress"] > 75, "Rating"].mean()


df.loc[df["Progress"] <= 10, "Rating"].mean() * 22/100 + \
df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24/100 + \
df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26/100 + \
df.loc[df["Progress"] > 75, "Rating"].mean() * 28/100


def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1/100 + \
           dataframe.loc[(dataframe["Progress"] > 10) & (dataframe["Progress"] <= 45), "Rating"].mean() * w2/100 + \
           dataframe.loc[(dataframe["Progress"] > 45) & (dataframe["Progress"] <= 75), "Rating"].mean() * w3/100 + \
           dataframe.loc[dataframe["Progress"] > 75, "Rating"].mean() * w4/100


user_based_weighted_average(df) #4.8002
user_based_weighted_average(df, 20,24,26,30)

####################
# Weighted Rating
####################

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + \
           user_based_weighted_average(dataframe) * user_w/100


course_weighted_rating(df) #4.7826
course_weighted_rating(df, 40,60)
course_weighted_rating(df, 60, 40)


###################################################
# Sorting Products
###################################################

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 50)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df = pd.read_csv("DS_Bootcamp_13/datasets/product_sorting.csv")
df.shape
df.head()

####################
# Sorting by Rating
####################

df.sort_values("rating", ascending=False).head()

####################
# Sorting by Comment Count or Purchase Count
####################

df = df.rename(columns={'commment_count': 'comment_count'})

df.sort_values("comment_count", ascending=False).head()
df.sort_values("purchase_count", ascending=False).head()

####################
# Sorting by Rating, Comment and Purchase
####################

df["comment_count_scaled"] = MinMaxScaler(feature_range=(1,5)).fit(df[["comment_count"]]).transform(df[["comment_count"]])
df.head()
df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1,5)).fit(df[["purchase_count"]]).transform(df[["purchase_count"]])
df.head()

(df["comment_count_scaled"] * 32/100 +
 df["purchase_count_scaled"] * 26/100 +
 df["rating"] * 42/100)

def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return dataframe["comment_count_scaled"] * w1 / 100 + \
           dataframe["purchase_count_scaled"] * w2 / 100 + \
           dataframe["rating"] * w3 / 100

weighted_sorting_score(df)

df["weighted_sorting_score"] = weighted_sorting_score(df)
df.head()

df.sort_values("weighted_sorting_score", ascending=False).head(10)

df[df["course_name"].str.contains("Veri Bilimi")]
df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False)

####################
# Bayesian Average Rating Score
####################

# Sorting Products with 5 Star Rated
# Sorting Products According to Distribution of 5 Stars Rating

import math
import scipy.stats as st


def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


df.head()
df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point", "2_point", "3_point", "4_point", "5_point"]]), axis=1)
df.head()

df.sort_values("weighted_sorting_score", ascending=False).head()
df.sort_values("bar_score", ascending=False).head()

df[df["course_name"].index.isin([5,1])]
df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending= False)

####################
# Hybrid Sorting: BAR Score + Diğer Faktorler
####################

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Bayesian Average Rating Score ***

# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR Score + Other Factors

def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point", "2_point", "3_point", "4_point", "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)
    return bar_score * bar_w/100 + wss_score * wss_w/100

df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(10)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False)


############################################
# Uygulama: IMDB Movie Scoring & Sorting
############################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 50)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df = pd.read_csv("DS_Bootcamp_13/datasets/movies_metadata.csv", low_memory=False)
df.head()

df = df[["title", "vote_average", "vote_count"]]
df.head()
df.shape

########################
# Vote Average'a Göre Sıralama
########################

df.sort_values("vote_average", ascending=False).head()

df["vote_count"].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T

df[df["vote_count"] > 400].sort_values("vote_average", ascending=False).head()

########################
# vote_average * vote_count
########################

df["vote_count_scaled"] = MinMaxScaler(feature_range=(1,5)).fit(df[["vote_count"]]).transform(df[["vote_count"]])
df.head()

df["average_count_score"] = df["vote_average"] * df["vote_count_scaled"]
df.head()

df.sort_values("average_count_score", ascending=False).head()

########################
# IMDB Weighted Rating
########################


# weighted_rating = (v/(v+M) * r) + (M/(v+M) * C)

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

# Film 1:
# r = 8
# M = 500
# v = 1000

# (1000 / (1000+500))*8 = 5.33


# Film 2:
# r = 8
# M = 500
# v = 3000

# (3000 / (3000+500))*8 = 6.85

# (1000 / (1000+500))*9.5 = 6.33

# Film 1:
# r = 8
# M = 500
# v = 1000

# Birinci bölüm:
# (1000 / (1000+500))*8 = 5.33

# İkinci bölüm:
# 500/(1000+500) * 7 = 2.33

# Toplam = 5.33 + 2.33 = 7.66


# Film 2:
# r = 8
# M = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000+500))*8 = 6.85

# İkinci bölüm:
# 500/(3000+500) * 7 = 1

# Toplam = 7.85

M = 2500
C = df["vote_average"].mean() #5.6182

def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + ( M / (v + M) * C)

df.sort_values("average_count_score", ascending=False).head(10)

weighted_rating(7.40000, 11444.0000, M, C)

weighted_rating(8.10000  , 14075.00000, M, C)

weighted_rating(8.50000, 8358.00000, M, C)

df["weighted_rating"] = weighted_rating(df["vote_average"], df["vote_count"], M, C)

df.sort_values("weighted_rating", ascending=False).head(10)

####################
# Bayesian Average Rating Score
####################

# 12481                                    The Dark Knight
# 314                             The Shawshank Redemption
# 2843                                          Fight Club
# 15480                                          Inception
# 292                                         Pulp Fiction

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

bayesian_average_rating([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])

df = pd.read_csv("DS_Bootcamp_13/datasets/imdb_ratings.csv")
df = df.iloc[0:,1:]
df.head()

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]]), axis=1)
df.sort_values("bar_score", ascending=False).head(10)

# Weighted Average Ratings
# IMDb publishes weighted vote averages rather than raw data averages.
# The simplest way to explain it is that although we accept and consider all votes received by users,
# not all votes have the same impact (or ‘weight’) on the final rating.

# When unusual voting activity is detected,
# an alternate weighting calculation may be applied in order to preserve the reliability of our system.
# To ensure that our rating mechanism remains effective,
# we do not disclose the exact method used to generate the rating.
#
# See also the complete FAQ for IMDb ratings.


############################################
# SORTING REVIEWS
############################################

import pandas as pd
import math
import scipy.stats as st

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

###################################################
# Up-Down Diff Score = (up ratings) − (down ratings)
###################################################

# Review 1: 600 up 400 down total 1000
# Review 2: 5500 up 4500 down total 10000

def score_up_down_diff(up, down):
    return up - down

score_up_down_diff(600,400)
score_up_down_diff(5500,4500)

###################################################
# Score = Average rating = (up ratings) / (all ratings)
###################################################

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

score_average_rating(600,400)
score_average_rating(5500,4500)

# Review 1: 2 up 0 down total 2
# Review 2: 100 up 1 down total 101

score_average_rating(2, 0)
score_average_rating(100, 1)

###################################################
# Wilson Lower Bound Score
###################################################

# 600-400
# 0.6
# 0.5 0.7
# 0.5

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

wilson_lower_bound(600,400)
wilson_lower_bound(5500,4500)

wilson_lower_bound(2,0)
wilson_lower_bound(100,1)

###################################################
# Case Study
###################################################

up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})
comments.head()

comments["score_pos_neg_diff"] = comments.apply(lambda x: score_up_down_diff(x["up"], x["down"]), axis=1)
comments.head(20)
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"],x["down"]), axis=1)
comments.head(20)
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"],x["down"]), axis=1)
comments.head(20)

comments.sort_values("wilson_lower_bound", ascending=False).head()

######################################################
# AB TESTING
######################################################

######################################################
# Temel İstatistik Kavramları
######################################################

# Population
# Sampling
# "Without a grounding in Statistics, a Data Scientist is a Data Lab Assistant."
# "The Future of AI Will Be About Less Data, Not More"

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 10)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

############################
# Sampling (Örnekleme)
############################

population = np.random.randint(0, 80, 10000)

population.mean()

np.random.seed(115)

orneklem = np.random.choice(a=population, size=100)

orneklem.mean()

np.random.seed(10)
orneklem1 = np.random.choice(a=population, size=100)
orneklem2 = np.random.choice(a=population, size=100)
orneklem3 = np.random.choice(a=population, size=100)
orneklem4 = np.random.choice(a=population, size=100)
orneklem5 = np.random.choice(a=population, size=100)
orneklem6 = np.random.choice(a=population, size=100)
orneklem7 = np.random.choice(a=population, size=100)
orneklem8 = np.random.choice(a=population, size=100)
orneklem9 = np.random.choice(a=population, size=100)
orneklem10 = np.random.choice(a=population, size=100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10

############################
# Descriptive Statistics (Betimsel İstatistikler)
############################

df = sns.load_dataset("tips")
df.describe().T

############################
# Confidence Intervals (Güven Aralıkları)
############################

### *** BOŞ DEĞERLER VARSA tconfint_mean() çalışmaz (nan, nan) döner *** ###

df = sns.load_dataset("tips")
df.describe().T

df.head()

# elle hesap

M = df["total_bill"].mean()
n = df["total_bill"].count()
s = df["total_bill"].std()
z = 1.96

alt_limit = M - (1.96 * (s / (np.sqrt(n))))
üst_limit = M + (1.96 * (s / (np.sqrt(n))))

sms.DescrStatsW(df["total_bill"]).tconfint_mean()
sms.DescrStatsW(df["tip"]).tconfint_mean()

df = sns.load_dataset("titanic")
df.describe().T

sms.DescrStatsW(df["age"].dropna()).tconfint_mean()
sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()

######################################################
# Correlation (Korelasyon)
######################################################

# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (bahşiş ve vergi dahil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker: grupta sigara içen var mı? (0=No, 1=Yes)
# day: gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: ne zaman? (0=Day, 1=Night)
# size: grupta kaç kişi var?

df = sns.load_dataset("tips")
df.head()

df["total_bill"] = df["total_bill"] - df["tip"]

df.plot.scatter("tip", "total_bill")
plt.show()

df["tip"].corr(df["total_bill"])

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği Varsayımı
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.


############################
# Uygulama 1: Sigara İçenler ile İçmeyenlerin Hesap Ortalamaları Arasında İst Ol An Fark var mı?
############################

df = sns.load_dataset("tips")
df.head()
df.groupby("smoker").agg({"total_bill": "mean"})

# ***** EKSİK DEĞER VAR MI BAK *****

############################
# 1. Hipotezi Kur
############################

# H0: M1 = M2
# H1: M1 != M2

############################
# 2. Varsayım Kontrolü
############################

# Normallik Varsayımı
# Varyans Homojenliği Varsayımı

############################
# Normallik Varsayımı
############################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

############################
# Varyans Homojenligi Varsayımı
############################

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# 3 ve 4. Hipotezin Uygulanması
############################

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

############################
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
############################

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
############################

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat %.4f, p-value %.4f" % (test_stat, pvalue))

############################
# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. var mıdır?
############################

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})

# 1. Hipotezleri kur:
# H0: M1  = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. Yoktur)
# H1: M1! = M2 (... vardır)


# 2. Varsayımları İncele

# Normallik varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır

test_stat, pvalue = shapiro(df.loc[df["sex"] =="female", "age"].dropna())
print("Test Stat %.4f, p-value %.4f" % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["sex"] =="male", "age"].dropna())
print("Test Stat %.4f, p-value %.4f" % (test_stat, pvalue))

# Varyans homojenliği
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat %.4f, p-value %.4f" % (test_stat, pvalue))

# Varsayımlar sağlanmadığı için nonparametrik

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat %.4f, p-value %.4f" % (test_stat, pvalue))

# 90 280


############################
# Uygulama 3: Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark var mıdır?
############################

df = pd.read_csv("DS_Bootcamp_13/datasets/diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"})

# 1. Hipotezleri kur
# H0: M1 = M2
# Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark Yoktur
# H1: M1 != M2
# .... vardır.

# 2. Varsayımları İncele

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"])
print("Test Stat %.4f, p-value %.4f" % (test_stat, pvalue))


test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"])
print("Test Stat %.4f, p-value %.4f" % (test_stat, pvalue))

# Normallik varsayımı sağlanmadığı için nonparametrik.

# Hipotez (H0: M1 = M2)

test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 0, "Age"],
                                 df.loc[df["Outcome"] == 1, "Age"])
print("Test Stat %.4f, p-value %.4f" % (test_stat, pvalue))


###################################################
# İş Problemi: Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı?
###################################################

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

df = pd.read_csv("DS_Bootcamp_13/datasets/course_reviews.csv")
df.head()

df[df["Progress"] > 75]["Rating"].mean()
df[df["Progress"] < 25]["Rating"].mean()

test_stat, pvalue = shapiro(df.loc[df["Progress"] > 75, "Rating"])
print("Test Stat %.4f, p-value %.4f" % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Progress"] < 25, "Rating"])
print("Test Stat %.4f, p-value %.4f" % (test_stat, pvalue))

test_stat, pvalue = mannwhitneyu(df.loc[df["Progress"] > 75, "Rating"],
                                 df.loc[df["Progress"] < 25, "Rating"])
print("Test Stat %.4f, p-value %.4f" % (test_stat, pvalue))


######################################################
# AB Testing (İki Örneklem Oran Testi)
######################################################

# H0: p1 = p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İst. Ol. Anlamlı Farklılık Yoktur.
# H1: p1 != p2
# ... vardır

basari_sayısı = np.array([300, 250])
gozlem_sayısı = np.array([1000,1100])

test_stat, pvalue = proportions_ztest(count=basari_sayısı, nobs=gozlem_sayısı)
print("Test Stat %.4f, p-value %.4f" % (test_stat, pvalue))

basari_sayısı / gozlem_sayısı

############################
# Uygulama: Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Farklılık var mıdır?
############################

# H0: p1 = p2
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark yoktur

# H1: p1 != p2
# .. vardır

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"survived": "mean"})

df.loc[df["sex"] == "female", "survived"].mean()
df.loc[df["sex"] == "male", "survived"].mean()

female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

female_number = df.loc[df["sex"] == "female"].count().max() # or ["sex"].shape[0]
male_number = df.loc[df["sex"] == "male"].count().max() # or ["sex"].shape[0]

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count], nobs=[female_number,male_number])
print("Test Stat %.4f, p-value %.4f" % (test_stat, pvalue))

######################################################
# ANOVA (Analysis of Variance)
######################################################

# İkiden fazla grup ortalamasını karşılaştırmak için kullanılır.

df = sns.load_dataset("tips")
df.head()

df.groupby("day").agg({"total_bill": "mean"})

# 1. Hipotezleri kur

# HO: m1 = m2 = m3 = m4
# Grup ortalamaları arasında fark yoktur.

# H1: .. fark vardır

# 2. Varsayım kontrolü

# Normallik varsayımı
# Varyans homojenliği varsayımı

# Varsayım sağlanıyorsa one way anova
# Varsayım sağlanmıyorsa kruskal

# H0: Normal dağılım varsayımı sağlanmaktadır.

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, "p-value %.4f" % pvalue)

# H0: Varyans homojenliği varsayımı sağlanmaktadır.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# 3. Hipotez testi ve p-value yorumu

# Hiç biri sağlamıyor.
df.groupby("day").agg({"total_bill": ["mean","median"]})

# HO: Grup ortalamaları arasında ist ol anl fark yoktur

# parametrik anova testi:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])

# Nonparametrik anova testi:
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])

# Grup ortalamaları birbirine eşit değil ama bu eşitsizlik hangisinden kaynaklanıyor:
from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())
# alfa değerini değiştirerek reject'i True olacak ilk iki grup kıyaslamasını gördük.
tukey = comparison.tukeyhsd(0.07)
print(tukey.summary())


############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

############################################
# 1. Veri Ön İşleme
############################################

# !pip install mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df_ = pd.read_excel("DS_Bootcamp_13/datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

# hata alınırsa
# pip install openpyxl
# engine="openpyxl"

df.head()
df.describe().T
df.isnull().sum()
df.shape

df = df[~df["Invoice"].str.contains("C", na=False)]
df.dropna(inplace=True)
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)

df.isnull().sum()
df.describe().T

############################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################
df.head()

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1


df_fr = df[df['Country'] == "France"]

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

df_fr.groupby(['Invoice', 'StockCode']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:15, 0:15]


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr)

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df_fr, 10120)


############################################
# 3. Birliktelik Kurallarının Çıkarılması
############################################

frequent_itemsets = apriori(df=fr_inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)]

check_id(df_fr, 21086)

rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)].sort_values("confidence", ascending=False)

############################################
# 4. Çalışmanın Scriptini Hazırlama
############################################

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

rules[(rules["support"] > 0.05) & (rules["confidence"] > 0.1) & (rules["lift"] > 5)]. \
sort_values("confidence", ascending=False)

############################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################

# Örnek:
# Kullanıcı örnek ürün id: 22492

product_id = 22492
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)

recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:3]

check_id(df, 22326)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)
arl_recommender(rules, 22492, 3)

#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda x: "%.4f" % x)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("DS_Bootcamp_13/datasets/movies_metadata.csv", low_memory=False)
df.head()
df.shape

df["overview"].head()

tfidf = TfidfVectorizer(stop_words="english")

df["overview"].isnull().sum()
df["overview"] = df["overview"].fillna(" ")

tfidf_matrix = tfidf.fit_transform(df["overview"])

tfidf_matrix.shape

df["title"].shape # film sayısı doğrulama

tfidf.get_feature_names_out()
len(tfidf.get_feature_names_out()) # eşsiz kelimeler

tfidf_matrix.toarray() # skorları matris olarak görmek için


#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim.shape
cosine_sim[1]

#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################

indices = pd.Series(df.index, index=df["title"])

indices.index.value_counts()

indices = indices[~indices.index.duplicated(keep="last")]

indices["Cinderella"]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]

similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"]) # sherlock holmes ile diğer tüm filmlerin benzerliği

movie_indices = similarity_scores.sort_values("score", ascending=False) # sherlock holmes sıralı score'lar
movie_indices[1:11] # ilki kendi diğer en benzer 10 film

#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda x: "%.4f" % x)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("DS_Bootcamp_13/datasets/movies_metadata.csv", low_memory=False)
df.head()
df.shape


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words="english")
    dataframe["overview"] = dataframe["overview"].fillna(" ")
    tfidf_matrix = tfidf.fit_transform(dataframe["overview"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri oluşturma
    indices = pd.Series(dataframe.index, index=dataframe["title"])
    indices = indices[~indices.index.duplicated(keep="last")]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a göre benzerlik skorlarını hesaplama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi hariç ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe["title"].iloc[movie_indices]


cosine_sim = calculate_cosine_sim(df)
content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


###########################################
# Item-Based Collaborative Filtering
###########################################

# Veri seti: https://grouplens.org/datasets/movielens/

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

######################################
# Adım 1: Veri Setinin Hazırlanması
######################################

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

movie = pd.read_csv("DS_Bootcamp_13/datasets/movie_lens_dataset/movie.csv")
rating = pd.read_csv("DS_Bootcamp_13/datasets/movie_lens_dataset/rating.csv")
df = movie.merge(rating, how="left", on="movieId")
df.head()
df.shape
df["title"].value_counts()
df["title"].nunique()

######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################

# 1000'den az yorum alan filmleri veri setinden kaldırıyoruz.

comment_counts = pd.DataFrame(df["title"].value_counts())

rare_movies = comment_counts[comment_counts["count"] < 1000].index

common_movies = df[~df["title"].isin(rare_movies)]

common_movies.shape
df.shape

common_movies["title"].nunique()
df["title"].nunique()

user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")
user_movie_df.head()

user_movie_df.shape
user_movie_df.columns

######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################

movie_name = "Matrix, The (1999)"

movie_name = user_movie_df[movie_name]

user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


movie_name = "Ocean's Twelve (2004)"

movie_name = user_movie_df[movie_name]

user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

movie_name = user_movie_df[movie_name]

user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]


check_film("Sherlock", user_movie_df) # letter sensitive


######################################
# Adım 4: Çalışma Scriptinin Hazırlanması
######################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("DS_Bootcamp_13/datasets/movie_lens_dataset/movie.csv")
    rating = pd.read_csv("DS_Bootcamp_13/datasets/movie_lens_dataset/rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movie = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")
    return user_movie_df


def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Matrix", user_movie_df) # letter sensitive



def item_based_recommender(movie_name, user_movie_df, number_of_recommendation=10):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(number_of_recommendation)


user_movie_df_2 = create_user_movie_df()

item_based_recommender("Matrix, The (1999)", user_movie_df_2, 5)


# film deneme
def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]


check_film("Holiday", user_movie_df)

item_based_recommender("Holiday, The (2006)", user_movie_df_2, 5)


############################################
# User-Based Collaborative Filtering
#############################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
# Adım 6: Çalışmanın Fonksiyonlaştırılması

#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

# movie = pd.read_csv("DS_Bootcamp_13/datasets/movie_lens_dataset/movie.csv")
# rating = pd.read_csv("DS_Bootcamp_13/datasets/movie_lens_dataset/rating.csv")
# df = movie.merge(rating, how="left", on="movieId")
# df.head()
# df.shape
# df["title"].nunique()

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('DS_Bootcamp_13/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('DS_Bootcamp_13/datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)


#############################################
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

random_user

user_movie_df.head()

random_user_df = user_movie_df[user_movie_df.index == random_user]

random_user_df.head()

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

len(movies_watched)

user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Schindler's List (1993)"]

user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Silence of the Lambs, The (1991)"]


#############################################
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

movies_watched_df = user_movie_df[movies_watched]

len(movies_watched_df.columns)

user_movie_count = movies_watched_df.T.notnull().sum()

user_movie_count = user_movie_count.reset_index()

user_movie_count.columns = ["userId", "movie_count"]

user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

len(user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False))

user_movie_count[user_movie_count["movie_count"] == 33].count()

users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

len(users_same_movies)


# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#############################################

# Bunun için 3 adım gerçekleştireceğiz:
# 1. Sinan ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)], random_user_df[movies_watched]])

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()

corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.index.names = ["user_id_1", "user_id_2"]

corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by="corr", ascending=False)

top_users.rename(columns={"user_id_2":"userId"}, inplace=True)

rating = pd.read_csv("DS_Bootcamp_13/datasets/movie_lens_dataset/rating.csv")

top_users_rating = top_users.merge(rating[["userId", "movieId", "rating"]], how="inner")

top_users_rating = top_users_rating[top_users_rating["userId"] != random_user]

#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#############################################

top_users_rating["weighted_rating"] = top_users_rating["corr"] * top_users_rating["rating"]

recommendation_df = top_users_rating.groupby("movieId").agg({"weighted_rating":"mean"})

recommendation_df = recommendation_df.reset_index()

recommendation_df.sort_values("weighted_rating", ascending=False)

recommendation_df[recommendation_df["weighted_rating"] > 3.5]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

movies = pd.read_csv("DS_Bootcamp_13/datasets/movie_lens_dataset/movie.csv")
movies_to_be_recommend.merge(movies[["movieId", "title"]])


#############################################
# Adım 6: Çalışmanın Fonksiyonlaştırılması
#############################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('DS_Bootcamp_13/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('DS_Bootcamp_13/datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    import pandas as pd
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    perc = len(movies_watched) * ratio / 100
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])

    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()

    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv('DS_Bootcamp_13/datasets/movie_lens_dataset/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('DS_Bootcamp_13/datasets/movie_lens_dataset/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])



random_user = int(pd.Series(user_movie_df.index).sample(1).values)
user_based_recommender(random_user, user_movie_df, cor_th=0.70, score=4)


#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################

# !pip install surprise
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv("DS_Bootcamp_13/datasets/movie_lens_dataset/movie.csv")
rating = pd.read_csv("DS_Bootcamp_13/datasets/movie_lens_dataset/rating.csv")
df = movie.merge(rating, how="left", on="movieId")
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df["movieId"].isin(movie_ids)]


sample_df["movieId"].nunique()
sample_df["movieId"].unique()
sample_df.shape
sample_df.head()

user_movie_df = sample_df.pivot_table(index=["userId"], columns=["title"], values=["rating"])

user_movie_df.shape

reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(sample_df[["userId", "movieId", "rating"]], reader)

##############################
# Adım 2: Modelleme
##############################

trainset, testset = train_test_split(data, test_size=0.25)
svd_model = SVD()
svd_model.fit(trainset)
predictions = svd_model.test(testset)
accuracy.rmse(predictions)


svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)

sample_df[sample_df["userId"] == 1]

##############################
# Adım 3: Model Tuning
##############################

param_grid = {"n_epochs":[5,10,20],
              "lr_all":[0.001,0.002,0.005,0.007]}

gs = GridSearchCV(SVD,
                  param_grid,
                  measures=["rmse","mae"],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)

gs.fit(data)

gs.best_score["rmse"]
gs.best_params["rmse"]

##############################
# Adım 4: Final Model ve Tahmin
##############################

dir(svd_model)
svd_model.n_epochs

svd_model = SVD(**gs.best_params["rmse"])

data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True)


#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

def load_application_train():
    data = pd.read_csv("DS_Bootcamp_13/datasets/application_train.csv")
    return data

df = load_application_train()
df.head()

def load():
    data = pd.read_csv("DS_Bootcamp_13/datasets/titanic.csv")
    return data

df = load()
df.head()

#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

#############################################
# Aykırı Değerleri Yakalama
#############################################

###################
# Grafik Teknikle Aykırı Değerler
###################

sns.boxplot(x=df["Age"])
plt.show()

###################
# Aykırı Değerler Nasıl Yakalanır?
###################

q1 = df["Age"].quantile(.25)
q2 = df["Age"].quantile(.5)
q3 = df["Age"].quantile(.75)

iqr = q3 - q1

up = q3 + (1.5 * iqr)
low = q1 - (1.5 * iqr)

df[(df["Age"] > up) | (df["Age"] < low)]

df[(df["Age"] > up) | (df["Age"] < low)].shape[0]

df[(df["Age"] > up) | (df["Age"] < low)].index


###################
# Aykırı Değer Var mı Yok mu?
###################

df[(df["Age"] > up) | (df["Age"] < low)].any(axis=None)
df[(df["Age"] < low)].any(axis=None)


# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok diye sorduk.

###################
# İşlemleri Fonksiyonlaştırmak
###################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartile_range = quartile3 - quartile1
    low_limit = quartile1 - (1.5 * interquartile_range)
    up_limit = quartile3 + (1.5 * interquartile_range)
    return low_limit, up_limit


low_limit, up_limit = outlier_thresholds(dataframe=df, col_name="Age")
df[(df["Age"] > up_limit) | (df["Age"] < low_limit)].any(axis=None)

low_limit, up_limit = outlier_thresholds(dataframe=df, col_name="Age", q1=.1, q3=.9)
df[(df["Age"] > up_limit) | (df["Age"] < low_limit)].any(axis=None)


def check_outlier(dataframe, col_name, q1=.25, q3=.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False


check_outlier(df, "Age")
check_outlier(df, "Age", q1=0.1, q3=.90)
check_outlier(df, "Fare")
check_outlier(df, "Fare", q1=.1, q3=.9)

###################
# grab_col_names
###################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if (dataframe[col].dtypes in ["int", "float"]) & (dataframe[col].nunique() < cat_th)]
    cat_but_car = [col for col in dataframe.columns if (str(dataframe[col].dtypes) in ["category", "object", "bool"]) & (dataframe[col].nunique() > car_th)]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

df = load()
df.head()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in ["PassengerId"]]

# id  ve date yakalamıyor

for col in num_cols:
    print(col, check_outlier(df, col))


dff = load_application_train()
dff.head()

cat_cols, num_cols, cat_but_car = grab_col_names(dff)


###################
# Aykırı Değerlerin Kendilerine Erişmek
###################

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].shape[0] > 10:
        print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].head())
    else:
        print(dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)])
    if index:
        outlier_index = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)].index
        return outlier_index

df.head()
grab_outliers(df, "Age")
grab_outliers(df,"Age", index=True)


#############################################
# Aykırı Değer Problemini Çözme
#############################################

###################
# Silme
###################

low, up = outlier_thresholds(df, "Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

df_without_outliers = remove_outlier(df, "Fare")

df_without_outliers.shape


df = load()
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if cat_but_car not in ["PassengerId"]]

for col in num_cols:
    new_df = remove_outlier(df, col)

new_df.shape

df.shape[0] - new_df.shape[0]

###################
# Baskılama Yöntemi (re-assignment with thresholds)
###################

low, up = outlier_thresholds(df, "Fare")

df[(df["Fare"] < low) | (df["Fare"] > up)]["Fare"]

df.loc[(df["Fare"] < low) | (df["Fare"] > up), "Fare"]


def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit
    dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit
    return dataframe


df = load()
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in ["PassengerId"]]

for col in num_cols:
    print(col, check_outlier(df, col))


for col in num_cols:
    replace_with_thresholds(df, col)


for col in num_cols:
    print(col, check_outlier(df, col))

###################
# Recap
###################

df=load()
df.shape
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", index=True)
remove_outlier(df, "Age").shape
replace_with_thresholds(df, "Age")
check_outlier(df, "Age")


#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################

df = sns.load_dataset("diamonds")
df.head()
df = df.select_dtypes(include=["int", "float"])
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))


low, up = outlier_thresholds(df, "carat")

df[(df["carat"] < low) | (df["carat"] > up)].shape
df.shape

low, up = outlier_thresholds(df, "depth")

df[(df["depth"] < low) | (df["depth"] > up)].shape

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
# df_scores = -df_scores
df_scores[0:5]

np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0,20], style=".-")
plt.show()

np.sort(df_scores)[3]
th = np.sort(df_scores)[3]

df[df_scores < th]
df[df_scores < th].shape
df.describe().T

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df.head()
df[df_scores < th].index
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)


#############################################
# Missing Values (Eksik Değerler)
#############################################

#############################################
# Eksik Değerlerin Yakalanması
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

def load_application_train():
    data = pd.read_csv("DS_Bootcamp_13/datasets/application_train.csv")
    return data

def load():
    data = pd.read_csv("DS_Bootcamp_13/datasets/titanic.csv")
    return data


df = load()

df.isnull().values.any()

df.isnull().sum()

df.notnull().sum()

df.isnull().sum().sum()
# or
df.isnull().values.sum()

df[df.isnull().any(axis=1)]

df[df.notnull().all(axis=1)]

df.isnull().sum().sort_values(ascending=False)

(df.isnull().sum()/df.shape[0] * 100).sort_values(ascending=False)

na_cols = [col for col in df.columns if df[col].isnull().any()]
# or
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().any()]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)
na_columns = missing_values_table(df, na_name=True)

#############################################
# Eksik Değer Problemini Çözme
#############################################

missing_values_table(df)

###################
# Çözüm 1: Hızlıca silmek
###################

df.shape
df.dropna().shape
#df.dropna(inplace=True)

###################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###################


df = load()
df.shape
df.isnull().sum()

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Age"] = df["Age"].fillna(0)

df["Age"].isnull().sum()

df = load()
df.shape
df.isnull().sum()

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtypes in ["int", "float"] else x, axis=0) # **** axis farklı bir kullanım ****

dff.isnull().sum().sort_values(ascending=False)


df["Embarked"].mode()[0]

dff = dff.apply(lambda x: x.fillna(x.mode()[0]) if (str(x.dtypes) in ["category", "object", "bool"]) & (x.nunique() < 10) else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

dff["Cabin"].nunique()

dff = df["Embarked"].fillna("missing")

dff.isnull().sum()

###################
# Kategorik Değişken Kırılımında Değer Atama
###################

df = load()

df.groupby("Sex").agg({"Age":"mean"})

df["Age"].mean()

df["Age"] = df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean"))

df["Age"].isnull().sum()

#or

df = load()

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df["Age"].isnull().sum()

#or

df = load()

df[df["Sex"] == "female"] = df[df["Sex"] == "female"].fillna(df.groupby("Sex")["Age"].mean()[0])

df[df["Sex"] == "male"] = df[df["Sex"] == "male"].fillna(df.groupby("Sex")["Age"].mean()[1])

df["Age"].isnull().sum()

#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma
#############################################

df = load()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if (dataframe[col].dtypes in ["int", "float"]) & (dataframe[col].nunique() < cat_th)]
    cat_but_car = [col for col in dataframe.columns if (str(dataframe[col].dtypes) in ["category", "object", "bool"]) & (dataframe[col].nunique() > car_th)]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in ["PassengerId"]]

dff = pd.get_dummies(df[cat_cols+num_cols], drop_first=True, dtype=int)
dff.head()

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]

df[df["Age"].isnull()]


#############################################
# Gelişmiş Analizler
#############################################

###################
# Eksik Veri Yapısının İncelenmesi
###################

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################

missing_values_table(df, na_name=False)

na_cols = missing_values_table(df, na_name=True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Survived", na_cols)

###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, na_name=True)
# sayısal değişkenleri direk median ile doldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtypes in ["int", "float"] else x, axis=0).isnull().sum()
# kategorik değişkenleri direk mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (str(x.dtypes) in ["category", "object", "bool"]) & (x.nunique() < 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurma
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# tahmine dayalı atama ile doldurma
missing_vs_target(df, "Survived", na_cols)


#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

#############################################
# Label Encoding & Binary Encoding
#############################################

df = load()
df.head()

df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]
# alfabetik ilkine 0 diğerine 1 veriyor

le.inverse_transform([0,1])
# 0 ne 1 ne görmek için


def label_encoder(dataframe, binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

binary_cols = [col for col in df.columns if (str(df[col].dtypes) in ["category", "object", "bool"]) & (df[col].nunique() == 2)]
# len(df[col].unique() ile yapılırsa ve eksik değer varsa sayı 3 çıkar
# df[col].nunique() ile eksik değerler sayılmaz ama label encoderdan geçirirken eksik değerleri de 2 olarak doldurur.
# ya 2'nin eksik değer olduğunu bileceğiz ya da öncesinde eksik değerlere müdahale edeceğiz.

for col in binary_cols:
    label_encoder(df, col)

df.head()

df = load_application_train()

binary_cols = [col for col in df.columns if (str(df[col].dtypes) in ["category", "object", "bool"]) & (df[col].nunique() == 2)]
len(binary_cols)

for col in binary_cols:
    label_encoder(df, col)

df[binary_cols].head()


#############################################
# One-Hot Encoding
#############################################

df = load()
df.head()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"])

pd.get_dummies(df, columns=["Embarked"], drop_first=True, dtype=int)

pd.get_dummies(df, columns=["Embarked"], drop_first=True, dtype=int, dummy_na=True)
# dummy_na eksik değere de sınıf oluşturur.

pd.get_dummies(df, columns=["Sex"], drop_first=True, dtype=int)


def one_hot_encoder(dataframe, categorical_cols, drop_first=True, dtype=int, dummy_na=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=dtype, dummy_na=dummy_na)
    return dataframe

df = load()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if (dataframe[col].dtypes in ["int", "float"]) & (dataframe[col].nunique() < cat_th)]
    cat_but_car = [col for col in dataframe.columns if (str(dataframe[col].dtypes) in ["category", "object", "bool"]) & (dataframe[col].nunique() > car_th)]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in ["PassengerId"]] # sayısal kolonlarla bi işimiz yok aslında

# bağımlı değişken encoder'a sokulmaz
cat_cols = [col for col in cat_cols if col not in ["Survived"]]

one_hot_encoder(df, cat_cols)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols)

df.head()
# kalıcı değişiklikler için df = one_hot_encoder(df, ohe_cols) diye atamamız lazım


#############################################
# Rare Encoding
#############################################

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

###################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
len(cat_cols)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)



for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col)
    else:
        cat_summary(df, col)

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################

df["NAME_INCOME_TYPE"].value_counts()
df.groupby("NAME_INCOME_TYPE").agg({"TARGET":"mean"})

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

#############################################
# 3. Rare encoder'ın yazılması.
#############################################

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if (str(temp_df[col].dtypes) in ["category", "object", "bool"])
                    & ((temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None))]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)


#############################################
# Feature Scaling (Özellik Ölçeklendirme)
#############################################

###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()
df.describe().T

###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################

rs = RobustScaler()
df["Age_robust_scaler"] = rs.fit_transform(df[["Age"]])
df.head()
df.describe().T

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T


age_cols = [col for col in df.columns if "Age" in col]


def num_summary(dataframe, col_name, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    print(dataframe[col_name].describe().T)

    if plot:
        dataframe[col_name].hist()
        plt.xlabel(col_name)
        plt.title(col_name)
        plt.show(block=True)


for col in age_cols:
    num_summary(df, col, plot=True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

df["Age_qcut"] = pd.qcut(df['Age'], 5)
df.head()


#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################

df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype("int")

df.groupby("NEW_CABIN_BOOL").agg({"Survived":"mean"})

from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],
                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))


df.loc[((df["SibSp"] + df["Parch"]) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df["SibSp"] + df["Parch"]) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived":"mean"})

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],
                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))


#############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################

df = load()
df.head()
df["Name"]

###################
# Letter Count
###################

df["NEW_NAME_COUNT"] = df["Name"].str.len()
df["NEW_NAME_COUNT"].sort_values(ascending=False)

###################
# Word Count
###################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
df["NEW_NAME_WORD_COUNT"].sort_values(ascending=False)

###################
# Özel Yapıları Yakalamak
###################

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df["NEW_NAME_DR"].unique()
df["NEW_NAME_DR"].sum()
df.groupby("NEW_NAME_DR").agg({"Survived":"mean"})

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_NAME_DR"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_NAME_DR"] == 0, "Survived"].sum()],
                                      nobs=[df.loc[df["NEW_NAME_DR"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_NAME_DR"] == 0, "Survived"].shape[0]])

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))


###################
# Regex ile Değişken Türetmek
###################

df["Name"]

df["NEW_TITLE"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

df["NEW_TITLE"]

df.groupby("NEW_TITLE").agg({"Survived":["count", "mean"],
                            "Age":["count", "mean"]})

#############################################
# Date Değişkenleri Üretmek
#############################################

import datetime as dt

dff =  pd.read_csv("DS_Bootcamp_13/datasets/course_reviews.csv")
dff.head()
dff.info()

dff['Timestamp'] = pd.to_datetime(dff['Timestamp'], format='%Y-%m-%d %H:%M:%S')

dff.info()

dff["year"] = dff["Timestamp"].dt.year

dff['month'] = dff['Timestamp'].dt.month

dff['day'] = dff['Timestamp'].dt.day

dff["day_name"] = dff["Timestamp"].dt.day_name()

dff["year_diff"] = date.today().year - dff["Timestamp"].dt.year

dff["month_diff"] = ((date.today().year - dff["Timestamp"].dt.year) * 12) + (date.today().month - dff["Timestamp"].dt.month)

#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################

df = load()
df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df["Sex"] == "male") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngmale"

df.loc[(df["Sex"] == "male") & (df["Age"] > 21) & (df["Age"] <= 50), "NEW_SEX_CAT"] = "maturemale"

df.loc[(df["Sex"] == "male") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniormale"

df.loc[(df["Sex"] == "female") & (df["Age"] <= 21), "NEW_SEX_CAT"] = "youngfemale"

df.loc[(df["Sex"] == "female") & (df["Age"] > 21) & (df["Age"] <= 50), "NEW_SEX_CAT"] = "maturefemale"

df.loc[(df["Sex"] == "female") & (df["Age"] > 50), "NEW_SEX_CAT"] = "seniorfemale"

df.groupby("NEW_SEX_CAT").agg({"Survived":"mean"})


#############################################
# Titanic Uçtan Uca Feature Engineering & Data Preprocessing
#############################################

df = load()
df.shape
df.head()

df.columns = [col.upper() for col in df.columns]


#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################


# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Değişkenleri ayırır.
    Parameters
    ----------
    dataframe : dataframe
        Veriseti
    cat_th : int, float
        Kategorik Değişken Tresshold
    car_th : int, float
        Kardinal Değişken Tresshold

    Returns
        cat_cols
        num_cols
        cat_but_car

    Notes
        cat_cols + num_cols + car_but_car = toplam değişken sayısı
    -------

    """
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in df.columns if (df[col].dtypes in ["int", "float"]) & (df[col].nunique() < cat_th)]
    cat_but_car = [col for col in df.columns if (str(df[col].dtypes) in ["category", "object"]) & (df[col].nunique() > car_th)]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables {dataframe.shape[1]}")
    print(f"cat_cols {len(cat_cols)}")
    print(f"num_cols {len(num_cols)}")
    print(f"cat_but_car {len(cat_but_car)}")
    print(f"num_but_cat {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]


#############################################
# 2. Outliers (Aykırı Değerler)
#############################################

def outlier_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    interquartile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartile_range
    low_limit = quartile1 - 1.5 * interquartile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1=.25, q3=.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df,col)

for col in num_cols:
    print(col, check_outlier(df, col))

#############################################
# 3. Missing Values (Eksik Değerler)
#############################################

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().any()]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

# age değişkenine bağlı oluşturulan değişkenler tekrar oluşturulur.

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (str(x.dtypes) in ["category", "object", "bool"]) & (x.nunique() <= 10) else x, axis=0)

missing_values_table(df)

#############################################
# 4. Label Encoding
#############################################

binary_cols = [col for col in df.columns if (str(df[col].dtypes) in ["category", "object", "bool"]) & (df[col].nunique() == 2)]

def label_encoder(dataframe, binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in binary_cols:
    df = label_encoder(df,col)

df.head()

#############################################
# 5. Rare Encoding
#############################################

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SURVIVED", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if (str(temp_df[col].dtypes) in ["category", "object", "bool"])
                    & ((temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None))]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

#############################################
# 6. One-Hot Encoding
#############################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True, dtype=int, dummy_na=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=dtype, dummy_na=dummy_na)
    return dataframe

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

df.drop(useless_cols, axis=1, inplace=True)

#############################################
# 7. Standart Scaler
#############################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape

#############################################
# 8. Model
#############################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# 0.7985074626865671

# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train, num=10)


#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# 0.7090909090909091
# 0.7985074626865671





######################################################
# Sales Prediction with Linear Regression
######################################################

######################################################
# Simple Linear Regression
######################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

df = pd.read_csv("DS_Bootcamp_13/datasets/advertising.csv")
df.head()
df.shape
df.describe().T
df.isnull().sum()

X = df[["TV"]]
y = df[["sales"]]

type(X)

##########################
# Model
##########################

reg_model = LinearRegression().fit(X, y)

# y_hat = b + w*x

# sabit (b - bias)
reg_model.intercept_[0]

# tv'nin katsayısı (w1)
reg_model.coef_[0][0]


##########################
# Tahmin
##########################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?

reg_model.intercept_[0] + reg_model.coef_[0][0] * 150

# 500 birimlik tv harcaması olsa ne kadar satış olur?

reg_model.intercept_[0] + reg_model.coef_[0][0] * 500

df.describe().T

# Modelin Görselleştirilmesi
g = sns.regplot(x=X, y=y, scatter_kws={"color": "b", "s" : 9},
                ci=False, color="r")
g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()

##########################
# Tahmin Başarısı
##########################

# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred) #10.512652915656757
y.mean()
y_pred.mean()
y.std()

# RMSE
np.sqrt(mean_squared_error(y, y_pred)) #3.2423221486546887
y.mean()
y.std()

# MAE
mean_absolute_error(y, y_pred) #2.549806038927486
y.mean()
y.std()

# R-KARE
reg_model.score(X, y) #0.611875050850071
# Veri setindeki bağımsız değişkenler bağımlı değişkeni %61 oranında açıklar

######################################################
# Multiple Linear Regression
######################################################

df = pd.read_csv("DS_Bootcamp_13/datasets/advertising.csv")

X = df.drop("sales", axis=1)
y = df[["sales"]]

##########################
# Model
##########################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

X_train.shape
y_train.shape

X_test.shape
y_test.shape

reg_model = LinearRegression().fit(X_train, y_train)

# sabit b - bias
reg_model.intercept_[0]

# coefficient (w - weights)
reg_model.coef_[0][0]
reg_model.coef_[0][1]
reg_model.coef_[0][2]

##########################
# Tahmin
##########################

# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# TV: 30
# radio: 10
# newspaper: 40

# 2.90
# 0.0468431 , 0.17854434, 0.00258619

# Sales = 2.90  + TV * 0.04 + radio * 0.17 + newspaper * 0.002

reg_model.intercept_[0] + 30*reg_model.coef_[0][0] + 10*reg_model.coef_[0][1] + 40*reg_model.coef_[0][2]

reg_model.intercept_[0] + df["TV"]*reg_model.coef_[0][0] + df["radio"]*reg_model.coef_[0][1] + df["newspaper"]*reg_model.coef_[0][2]

yeni_veri = [[30], [10], [40]]
yeni_veri = pd.DataFrame(yeni_veri).T
reg_model.predict(yeni_veri)

##########################
# Tahmin Başarısını Değerlendirme
##########################

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1.73

# Train R-KARE
reg_model.score(X_train, y_pred) #1 overfitting

# Train MAE
mean_squared_error(y_train, y_pred)

# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1.41

# Test R-KARE
reg_model.score(X_test, y_pred) #1 overfitting

# Test MAE
mean_squared_error(y_test, y_pred)

# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))
# 1.69

# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1.72

######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################

# Cost function MSE
def cost_function(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):

    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                   cost_function(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost_function(Y, b, w, X)
        cost_history.append(mse)


        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))


    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost_function(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("DS_Bootcamp_13/datasets/advertising.csv")
X = df["radio"]
Y = df["sales"]

# hyperparameters
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 100000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)


######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# İş Problemi:

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup
# olmadıklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirebilir misiniz?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Değişkenler
# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz.
# BloodPressure: Kan basıncı.
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_validate


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

######################################################
# Exploratory Data Analysis
######################################################

df = pd.read_csv("DS_Bootcamp_13/datasets/diabetes.csv")
df.head()
df.shape

##########################
# Target'ın Analizi
##########################

df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.show()

100 * df["Outcome"].value_counts() / len(df)

##########################
# Feature'ların Analizi
##########################

df.describe().T

df["BloodPressure"].hist(bins=20)
plt.xlabel("BloodPressure")
plt.show()

def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)

for col in df.columns:
    plot_numerical_col(df, col)

cols = [col for col in df.columns if "Outcome" not in col]

for col in cols:
    plot_numerical_col(df, col)

##########################
# Target vs Features
##########################

df.groupby("Outcome").agg({"Pregnancies":"mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n\n")


for col in cols:
    target_summary_with_num(df, "Outcome", col)

######################################################
# Data Preprocessing (Veri Ön İşleme)
######################################################

df.shape
df.head()

df.isnull().sum()
df.describe().T

for col in cols:
    print(col, check_outlier(df, col))

replace_with_thresholds(df, "Insulin")

for col in cols:
    print(col, check_outlier(df, col))

df.dtypes

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()
df.describe().T

######################################################
# Model & Prediction
######################################################

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)

y_pred[0:10]
y[0:10]

accuracy_score(y, y_pred)
precision_score(y, y_pred)
recall_score(y, y_pred)
f1_score(y, y_pred)
confusion_matrix(y, y_pred) # grafiksiz kötü
classification_repor(y, y_pred) # tüm skorlar tek yerde
# manuel skor hesaplama
accuracy1 = (446+156)/(446+54+112+156)
precision1 = (156/(156+54))
recall = (156/(156+112))
f1_score1 = (2*precision1*recall)/(precision1+recall)

######################################################
# Model Evaluation
######################################################

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))


# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1] # ikinci sütun (1) 1 olma ihtimali
roc_auc_score(y, y_prob)
# 0.83939


######################################################
# Model Validation: Holdout
######################################################

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63

RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)
# 0.87

######################################################
# Model Validation: 5-Fold Cross Validation
######################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63


cv_results['test_accuracy'].mean()
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.7192

cv_results['test_recall'].mean()
# Recall: 0.5747

cv_results['test_f1'].mean()
# F1-score: 0.6371

cv_results['test_roc_auc'].mean()
# AUC: 0.8327

######################################################
# Prediction for A New Observation
######################################################

X.columns

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)
random_user
df.iloc[195]



################################################
# KNN
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("DS_Bootcamp_13/datasets/diabetes.csv")
df.head()
df.shape
df.isnull().sum()
df.describe().T
df["Outcome"].value_counts()

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

df.dtypes

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns=X.columns)

# ya da direk aşağıdaki yöntemle hepsi sayısal olan bağımsız değişkenleri standartlaştırırız.
cols = [col for col in df.columns if "Outcome" not in col]

for col in cols:
    df[col] = StandardScaler().fit_transform(df[[col]])

df.head()

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)


################################################
# 3. Modeling & Prediction
################################################

knn_model = KNeighborsClassifier(n_neighbors=5).fit(X, y)

random_user = X.sample(1, random_state=45)

knn_model.predict(random_user)

################################################
# 4. Model Evaluation
################################################

# Confusion matrix için y_pred
y_pred = knn_model.predict(X)

# AUC için y_prob
y_prob = knn_model.predict_proba(X)[:,1]

print(classification_report(y, y_pred))
# acc 0.83
# f1 0.74

# AUC
roc_auc_score(y, y_prob)
# auc 0.90

cv_result = cross_validate(knn_model,
                           X,
                           y,
                           cv=5,
                           scoring=["accuracy", "f1", "roc_auc"])


cv_result["test_accuracy"].mean()
cv_result["test_f1"].mean()
cv_result["test_roc_auc"].mean()
# acc 0.73
# f1 0.59
# auc 0.78

# 1. Örnek boyutu arttıralabilir.
# 2. Veri ön işleme
# 3. Özellik mühendisliği
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()

################################################
# 5. Hyperparameter Optimization
################################################

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1).fit(X, y)

knn_gs_best.best_params_

################################################
# 6. Final Model
################################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
# acc 0.76
# f1 0.61
# auc 0.81

random_user = X.sample(1) #index = 95

knn_final.predict(random_user) #Outcome = 0

df[df.index == 95]["Outcome"] #Outcome = 0



################################################
# Decision Tree Classification: CART
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling using CART
# 4. Hyperparameter Optimization with GridSearchCV
# 5. Final Model
# 6. Feature Importance
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
# 8. Visualizing the Decision Tree
# 9. Extracting Decision Rules
# 10. Extracting Python/SQL/Excel Codes of Decision Rules
# 11. Prediction using Python Codes
# 12. Saving and Loading Model


# pip install pydotplus
# pip install skompiler
# pip install astor
# pip install joblib
# conda install graphviz

import warnings
#pip install joblib
import joblib
#pip install pydotplus
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
#pip install skompiler
from skompiler import skompile
#pip install astor
import graphviz

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

################################################
# 1. Exploratory Data Analysis
################################################

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

################################################
# 3. Modeling using CART
################################################

df = pd.read_csv("DS_Bootcamp_13/datasets/diabetes.csv")
df.head()

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

cart_model = DecisionTreeClassifier(random_state=1)
cart_model.fit(X, y)

# Confusion matrix için y_pred
y_pred = cart_model.predict(X)

# AUC için y_prob
y_prob = cart_model.predict_proba(X)[:,1]

# Confusion matrax
print(classification_report(y, y_pred))

# AUC Score
roc_auc_score(y, y_prob)

#####################
# Holdout Yöntemi ile Başarı Değerlendirme
#####################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=45)

cart_model = DecisionTreeClassifier(random_state=17)
cart_model.fit(X_train, y_train)

# Train Hatası
y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:,1]
print(classification_report(y_train, y_prob))
roc_auc_score(y_train, y_prob)

#  Test Hatası
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_prob))
roc_auc_score(y_test, y_prob)

#####################
# CV ile Başarı Değerlendirme
#####################

cart_model = DecisionTreeClassifier(random_state=17)
cart_model.fit(X, y)

cv_results = cross_validate(cart_model,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
# 0.7058568882098294
cv_results["test_f1"].mean()
# 0.5710621194523633
cv_results["test_roc_auc"].mean()
# 0.6719440950384347

################################################
# 4. Hyperparameter Optimization with GridSearchCV
################################################

cart_model.get_params()

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              scoring="accuracy", #default
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X, y)

cart_best_grid.best_params_

cart_best_grid.best_score_

random = X.sample(1, random_state=45)

cart_best_grid.predict(random) #cart_best_grid de bir model ama final modeli kuruyoruz.

################################################
# 5. Final Model
################################################

cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
cart_final.get_params()
#ya da set_params ile (önerilen)
cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y) #bu aşamada criterion="entropy" de yapabilirsin

cv_results = cross_validate(cart_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])


cv_results["test_accuracy"].mean()
# 0.7500806383159324
cv_results["test_f1"].mean()
# 0.6161442661552334
cv_results["test_roc_auc"].mean()
# 0.8000573025856046

################################################
# 6. Feature Importance
################################################

cart_final.feature_importances_

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(cart_final, X, num=5)
# görüntüyü save=True ile kaydedebiliyoruz
# plot_importance(cart_final, X, num=5, save=True)

################################################
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
################################################

train_score, test_score = validation_curve(cart_final, X, y,
                                           param_name="max_depth",
                                           param_range=range(1,11),
                                           scoring="roc_auc",
                                           cv=10) # her bir parametre değeri için bir array olur. Her arrayin eleman sayısı da cv sayısıdır.

mean_train_score = np.mean(train_score, axis=1) # her bir parametrenin score'ları
mean_test_score = np.mean(test_score, axis=1) # her bir parametrenin score'ları


plt.plot(range(1, 11), mean_train_score,
         label="Training Score", color='b')
plt.plot(range(1, 11), mean_test_score,
         label="Validation Score", color='g')
plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show()

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

val_curve_params(cart_model, X, y, "max_depth", range(1,11), "f1", 10)

cart_val_params = [["max_depth", range(1,11)], ["min_samples_split", range(2,20)]]

for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, param_name=cart_val_params[i][0], param_range=cart_val_params[i][1], scoring="roc_auc", cv=10)

# her bir param için tek tek eğrilere bakıyoruz ama en iyisi ile değiştirmiyoruz.
# zaten en iyi param'ları biz GridSearchCV ile tüm paramlar ile en iyi sonuçları elde ettik.
# min_sample_split grafikte en iyi 3 çıkabilir ama diğerleriyle en iyi kombinasyonu 5 çıkıyor.
# grafiğe hiç bakmasak da olur. Sadece tek bir param'ı değerlendirme için bakılır.

################################################
# 8. Visualizing the Decision Tree
################################################

# conda install graphviz
# import graphviz

def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)

tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final_entropy.png")

cart_final.get_params()

################################################
# 9. Extracting Decision Rules
################################################

tree_rules = export_text(cart_final, feature_names=list(X.columns))
print(tree_rules)

################################################
# 10. Extracting Python Codes of Decision Rules
################################################

# sklearn '0.23.1' versiyonu ile yapılabilir.
# pip install scikit-learn==0.23.1

print(skompile(cart_final.predict).to('python/code'))

print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))

print(skompile(cart_final.predict).to('excel'))

################################################
# 11. Prediction using Python Codes
################################################

def predict_with_rules(x):
    return ((((((0 if x[6] <= 0.671999990940094 else 1 if x[6] <= 0.6864999830722809 else
        0) if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else ((1 if x[5
        ] <= 32.45000076293945 else 1 if x[3] <= 10.5 else 0) if x[2] <= 53.0 else
        ((0 if x[1] <= 111.5 else 0 if x[2] <= 72.0 else 1 if x[3] <= 31.0 else
        0) if x[2] <= 82.5 else 1) if x[4] <= 36.5 else 0) if x[6] <=
        0.5005000084638596 else (0 if x[1] <= 88.5 else (((0 if x[0] <= 1.0 else
        1) if x[1] <= 98.5 else 1) if x[6] <= 0.9269999861717224 else 0) if x[1
        ] <= 116.0 else 0 if x[4] <= 166.0 else 1) if x[2] <= 69.0 else ((0 if
        x[2] <= 79.0 else 0 if x[1] <= 104.5 else 1) if x[3] <= 5.5 else 0) if
        x[6] <= 1.098000019788742 else 1) if x[5] <= 45.39999961853027 else 0 if
        x[7] <= 22.5 else 1) if x[7] <= 28.5 else (1 if x[5] <=
        9.649999618530273 else 0) if x[5] <= 26.350000381469727 else (1 if x[1] <=
        28.5 else ((0 if x[0] <= 11.5 else 1 if x[5] <= 31.25 else 0) if x[1] <=
        94.5 else (1 if x[5] <= 36.19999885559082 else 0) if x[1] <= 97.5 else
        0) if x[6] <= 0.7960000038146973 else 0 if x[0] <= 3.0 else (1 if x[6] <=
        0.9614999890327454 else 0) if x[3] <= 20.0 else 1) if x[1] <= 99.5 else
        ((1 if x[5] <= 27.649999618530273 else 0 if x[0] <= 5.5 else (((1 if x[
        0] <= 7.0 else 0) if x[1] <= 103.5 else 0) if x[1] <= 118.5 else 1) if
        x[0] <= 9.0 else 0) if x[6] <= 0.19999999552965164 else ((0 if x[5] <=
        36.14999961853027 else 1) if x[1] <= 113.0 else 1) if x[0] <= 1.5 else
        (1 if x[6] <= 0.3620000034570694 else 1 if x[5] <= 30.050000190734863 else
        0) if x[2] <= 67.0 else (((0 if x[6] <= 0.2524999976158142 else 1) if x
        [1] <= 120.0 else 1 if x[6] <= 0.23899999260902405 else 1 if x[7] <=
        30.5 else 0) if x[2] <= 83.0 else 0) if x[5] <= 34.45000076293945 else
        1 if x[1] <= 101.0 else 0 if x[5] <= 43.10000038146973 else 1) if x[6] <=
        0.5609999895095825 else ((0 if x[7] <= 34.5 else 1 if x[5] <=
        33.14999961853027 else 0) if x[4] <= 120.5 else (1 if x[3] <= 47.5 else
        0) if x[4] <= 225.0 else 0) if x[0] <= 6.5 else 1) if x[1] <= 127.5 else
        (((((1 if x[1] <= 129.5 else ((1 if x[6] <= 0.5444999933242798 else 0) if
        x[2] <= 56.0 else 0) if x[2] <= 71.0 else 1) if x[2] <= 73.0 else 0) if
        x[5] <= 28.149999618530273 else (1 if x[1] <= 135.0 else 0) if x[3] <=
        21.0 else 1) if x[4] <= 132.5 else 0) if x[1] <= 145.5 else 0 if x[7] <=
        25.5 else ((0 if x[1] <= 151.0 else 1) if x[5] <= 27.09999942779541 else
        ((1 if x[0] <= 6.5 else 0) if x[6] <= 0.3974999934434891 else 0) if x[2
        ] <= 82.0 else 0) if x[7] <= 61.0 else 0) if x[5] <= 29.949999809265137
         else ((1 if x[2] <= 61.0 else (((((0 if x[6] <= 0.18299999833106995 else
        1) if x[0] <= 0.5 else 1 if x[5] <= 32.45000076293945 else 0) if x[2] <=
        73.0 else 0) if x[0] <= 4.5 else 1 if x[6] <= 0.6169999837875366 else 0
        ) if x[6] <= 1.1414999961853027 else 1) if x[5] <= 41.79999923706055 else
        1 if x[6] <= 0.37299999594688416 else 1 if x[1] <= 142.5 else 0) if x[7
        ] <= 30.5 else (((1 if x[6] <= 0.13649999350309372 else 0 if x[5] <=
        32.45000076293945 else 1 if x[5] <= 33.05000114440918 else (0 if x[6] <=
        0.25599999725818634 else (0 if x[1] <= 130.5 else 1) if x[0] <= 8.5 else
        0) if x[0] <= 13.5 else 1) if x[2] <= 92.0 else 1) if x[5] <=
        45.54999923706055 else 1) if x[6] <= 0.4294999986886978 else (1 if x[5] <=
        40.05000114440918 else 0 if x[5] <= 40.89999961853027 else 1) if x[4] <=
        333.5 else 1 if x[2] <= 64.0 else 0) if x[1] <= 157.5 else ((((1 if x[7
        ] <= 25.5 else 0 if x[4] <= 87.5 else 1 if x[5] <= 45.60000038146973 else
        0) if x[7] <= 37.5 else 1 if x[7] <= 56.5 else 0 if x[6] <=
        0.22100000083446503 else 1) if x[6] <= 0.28849999606609344 else 0) if x
        [6] <= 0.3004999905824661 else 1 if x[7] <= 44.0 else (0 if x[7] <=
        51.0 else 1 if x[6] <= 1.1565000414848328 else 0) if x[0] <= 6.5 else 1
        ) if x[4] <= 629.5 else 1 if x[6] <= 0.4124999940395355 else 0)

X.columns

x = [12, 13, 20, 23, 4, 55, 12, 7]

predict_with_rules(x)

x = [6, 148, 70, 35, 0, 30, 0.62, 50]

predict_with_rules(x)

################################################
# 12. Saving and Loading Model
################################################

joblib.dump(cart_final, "cart_final.pkl")

cart_model_from_disc = joblib.load("cart_final.pkl")

x = [12, 13, 20, 23, 4, 55, 12, 7]

cart_model_from_disc.predict(pd.DataFrame(x).T)


################################################
# Random Forests, GBM, XGBoost, LightGBM, CatBoost
################################################

# !pip install scikit-learn==1.5.2          # xgboost için eski versiyon yüklenecek

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# !pip install catboost
# !pip install xgboost ########### eski versiyon yükle
# !pip install lightgbm
# !pip install scikit-learn

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("DS_Bootcamp_13/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)

rf_model.get_params()

cv_results = cross_validate(rf_model,
                            X,
                            y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
# 0.7525803144224197
cv_results["test_f1"].mean()
# 0.6165191330554752
cv_results["test_roc_auc"].mean()
# 0.8238418803418803

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model,
             rf_params,
             cv=5,
             scoring="roc_auc",
             n_jobs=-1,
             verbose=True).fit(X,y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final,
                            X,
                            y,
                            cv=10,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
# 0.768198906356801
cv_results["test_f1"].mean()
# 0.6265211688033927
cv_results["test_roc_auc"].mean()
# 0.8368176638176639

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")

################################################
# GBM
################################################

gbm_model = GradientBoostingClassifier(random_state=17)

gbm_model.get_params()

cv_results = cross_validate(gbm_model,
                           X,
                           y,
                           cv=5,
                           scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
# 0.7591715474068416
cv_results["test_f1"].mean()
# 0.634235802826363
cv_results["test_roc_auc"].mean()
# 0.8260164220824597

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model,
                             gbm_params,
                             cv= 5,
                             scoring="roc_auc",
                             n_jobs=-1,
                             verbose=True).fit(X,y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(gbm_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
# 0.7708853238265003
cv_results["test_f1"].mean()
# 0.6468931089325088
cv_results["test_roc_auc"].mean()
# 0.8393431167016073

################################################
# XGBoost
################################################

xgboost_model = XGBClassifier(random_state=17)

cv_results = cross_validate(xgboost_model,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
# 0.7409557762498938
cv_results["test_f1"].mean()
# 0.6180796532975465
cv_results["test_roc_auc"].mean()
# 0.7934919636617749

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 500, 1000],
                  "colsample_bytree": [0.7, 1]}

xgboost_best_grid = GridSearchCV(xgboost_model,
                                 xgboost_params,
                                 cv=5,
                                 n_jobs=-1,
                                 verbose=True).fit(X,y)

xgboost_best_grid.best_params_

xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X,y)

cv_results = cross_validate(xgboost_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
# 0.7591715474068416
cv_results["test_f1"].mean()
# 0.6455592036255984
cv_results["test_roc_auc"].mean()
# 0.8141781970649895

################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17, verbose=-1)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7474492827434004
cv_results['test_f1'].mean()
# 0.624110522144179
cv_results['test_roc_auc'].mean()
# 0.7990293501048218

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model,
                              lgbm_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7643578643578645
cv_results['test_f1'].mean()
# 0.6372062920577772
cv_results['test_roc_auc'].mean()
# 0.8147491264849755

# Hiperparametre yeni değerlerle
lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
               "n_estimators": [200, 300, 350, 400],
               "colsample_bytree": [0.9, 0.8, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model,
                              lgbm_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7643833290892115
cv_results['test_f1'].mean()
# 0.6193071162618689
cv_results['test_roc_auc'].mean()
# 0.8227931516422082

# Hiperparametre optimizasyonu sadece n_estimators için.
lgbm_model = LGBMClassifier(random_state=17, colsample_bytree=0.9, learning_rate=0.01)

lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 8000, 9000, 10000]}

lgbm_best_grid = GridSearchCV(lgbm_model,
                              lgbm_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=False).fit(X, y) # verbose=False yap

lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7643833290892115
cv_results['test_f1'].mean()
# 0.6193071162618689
cv_results['test_roc_auc'].mean()
# 0.8227931516422082

################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

cv_results = cross_validate(catboost_model,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7735251676428148
cv_results['test_f1'].mean()
# 0.6502723851348231
cv_results['test_roc_auc'].mean()
# 0.8378923829489867

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model,
                                  catboost_params,
                                  cv=5,
                                  n_jobs=-1,
                                  verbose=True).fit(X, y)

catboost_best_grid.best_params_

catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7695611577964518
cv_results['test_f1'].mean()
# 0.6308243414435932
cv_results['test_roc_auc'].mean()
# 0.8428623340321453

################################################
# Feature Importance
################################################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(xgboost_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)

################################
# Hyperparameter Optimization with RandomSearchCV (BONUS)
################################

rf_model = RandomForestClassifier(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}

rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=100,  # denenecek parametre sayısı
                               cv=3,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1).fit(X, y)

rf_random.best_params_

rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_random_final,
                            X,
                            y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7669977081741788
cv_results['test_f1'].mean()
# 0.6297429221187509
cv_results['test_roc_auc'].mean()
# 0.8350363382250174

################################
# Analyzing Model Complexity with Learning Curves (BONUS)
################################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]

rf_model = RandomForestClassifier(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

rf_val_params[0][1]


################################
# Imbalanced Dataset
################################

# Gerekli kütüphaneler
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc,rcParams
import itertools

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Veri setinin okutulması
df = pd.read_csv("DS_Bootcamp_13/datasets/creditcard.csv")
df.head()

# Veri setindeki değişken ve gözlem sayısı
print("Gözlem sayısı : ", len(df))
print("Değişken sayısı : ",  len(df.columns))

# veri setindeki değişkenlerin tiplerini ve boş değer içerip içermediğini gözlemlemek istiyoruz
df.info()

# 1 sınıfının veri setinde bulunma oranı %0.2, 0 sınıfının ise %99.8
f,ax=plt.subplots(1,2,figsize=(18,8))
df['Class'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('dağılım')
ax[0].set_ylabel('')
sns.countplot(data=df, x=df["Class"], ax=ax[1])
ax[1].set_title('Class')
plt.show(block=True)

# Time ve Amount değişkenlerini standartlaştırma
rob_scaler = RobustScaler()
df['Amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))
df.head()
df.describe().T

# Hold out yöntemi uygulayıp veri setini eğitim ve test olarak ikiye ayırıyoruz.(%80,%20)
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123456)

# modelin tanımlanıp, eğitilmesi ve başarı skoru
model = LogisticRegression(random_state=123456)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.3f"%(accuracy))


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.rcParams.update({'font.size': 19})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontdict={'size': '16'})
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12, color="blue")
    plt.yticks(tick_marks, classes, fontsize=12, color="blue")
    rc('font', weight='bold')
    fmt = '.1f'
    thresh = cm.max()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="red")

    plt.ylabel('True label', fontdict={'size': '16'})
    plt.xlabel('Predicted label', fontdict={'size': '16'})
    plt.tight_layout()

plot_confusion_matrix(confusion_matrix(y_test, y_pred=y_pred), classes=['Non Fraud','Fraud'], title='Confusion matrix')


#sınıflandırma raporu
print(classification_report(y_test, y_pred))


# Auc Roc Curve
def generate_auc_roc_curve(clf, X_test):
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr)
    plt.show(block=True)
    pass

generate_auc_roc_curve(model, X_test)

y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC ROC Curve with Area Under the curve = %.3f"%auc)


# random oversampling önce eğitim setindeki sınıf sayısı
y_train.value_counts()


# RandomOver Sampling uygulanması (Eğitim setine uygulanıyor)

# !pip install imbalanced-learn
# pip show imbalanced-learn
# pip show scikit-learn


from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy='minority')
X_randomover, y_randomover = oversample.fit_resample(X_train, y_train)


# random oversampling den sonra eğitim setinin sınıf sayısı
y_randomover.value_counts()

# modelin eğitilmesi ve başarı oranı
model.fit(X_randomover, y_randomover)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.3f%%" % (accuracy))

plot_confusion_matrix(confusion_matrix(y_test, y_pred=y_pred), classes=['Non Fraud','Fraud'], title='Confusion matrix')

#sınıflandırma raporu
print(classification_report(y_test, y_pred))

# smote dan önce eğitim setindeki sınıf sayısı
y_train.value_counts()

# Smote uygulanması (Eğitim setine uygulanıyor)
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X_train, y_train)

# smote dan sonra eğitim setinin sınıf sayısı
y_smote.value_counts()

# modelin eğitilmesi ve başarı oranı
model.fit(X_smote, y_smote)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.3f%%" % (accuracy))


plot_confusion_matrix(confusion_matrix(y_test, y_pred=y_pred), classes=['Non Fraud','Fraud'],
                      title='Confusion matrix')


#sınıflandırma raporu
print(classification_report(y_test, y_pred))


# random undersampling den önce eğitim setindeki sınıf sayısı
y_train.value_counts()


from imblearn.under_sampling import RandomUnderSampler
# transform the dataset
ranUnSample = RandomUnderSampler()
X_ranUnSample, y_ranUnSample = ranUnSample.fit_resample(X_train, y_train)

# Random undersampling sonra
y_ranUnSample.value_counts()
"""
NearMiss Undersampling:
•	Bilgi kaybını önler.
•	KNN algoritmasına dayanır.
•	Çoğunluk sınıfına ait örneklerin azınlık sınıfına ait örneklerle olan uzaklığı hesaplanır.
•	Belirtilen k değerine göre uzaklığı kısa olan örnekler korunur.
Undersampling (Tomek links):
Farklı sınıflara ait en yakın iki örneğin arasındaki çoğunluk sınıfının örnekleri kaldırılarak, iki sınıf arasındaki boşluk arttırılır.
Undersampling (Cluster Centroids):
Önemsiz örneklerin veri setinden çıkarılmasıdır. Örneğin önemli veya önemsiz olduğu kümelemeyle belirlenir.
Undersampling ve Oversampling tekniklerinin bir araya gelmesiyle daha dengeli veri setleri oluşturulabilinir.
Diğer Yöntemler
•	Daha fazla veri toplamak,
•	Sınıflandırma modellerinde bulunan “class_weight” parametresi kullanılarak azınlık ve çoğunluk sınıflarından eşit şekilde öğrenebilen model yaratılması,
•	Tek bir modele değil , diğer modellerdeki performanslara da bakılması,
•	Daha farklı bir yaklaşım uygulanıp Anomaly detection veya Change detection yapmak
gibi yöntemlerle de dengesiz veri setiyle başa çıkılır.
Hangi yöntemin en iyi sonuç vereceği elimizdeki veri setine bağlıdır. Yöntemler denenerek veri setine en uygun olanın seçilmesi en iyi sonucu sağlar diyebiliriz.

"""


################################
# Unsupervised Learning
################################

# pip install yellowbrick

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV

################################
# K-Means
# Hierarchical Clustering
# Principal Component Analysis
################################


################################
# K-Means
################################

df = pd.read_csv("DS_Bootcamp_13/datasets/USArrests.csv", index_col=0)
df.head()
df.isnull().sum()
df.info()
df.describe().T

sc = MinMaxScaler((0,1))
df = sc.fit_transform(df)
type(df)
df[0:5]

kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_

################################
# Optimum Küme Sayısının Belirlenmesi
################################

kmeans = KMeans()
ssd = []
K = range(1,30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSD/SSR")
plt.title("Optimum Küme Sayısı İçin Elbow Yöntemi")
plt.show(block=True)

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2,20))
elbow.fit(df)
plt.show(block=True)

elbow.elbow_value_

################################
# Final Cluster'ların Oluşturulması
################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5]

clusters_kmeans = kmeans.labels_

df = pd.read_csv("DS_Bootcamp_13/datasets/USArrests.csv", index_col=0)

df["kmeans_clusters"] = clusters_kmeans

df.head()

df["kmeans_clusters"] = df["kmeans_clusters"] + 1

df.head()

df[df["kmeans_clusters"] == 5]

df.groupby("kmeans_clusters").agg(["count", "mean", "median"])

df.to_csv("kmeans_clusters.csv")


################################
# Hierarchical Clustering
################################

df = pd.read_csv("DS_Bootcamp_13/datasets/USArrests.csv", index_col=0)
df.head()

sc = MinMaxScaler((0,1))
df = sc.fit_transform(df)
type(df)

hc_average = linkage(df, "average")

# her bir küme grafiğin altında görünüyor.
plt.figure(figsize=(19,5))
plt.title("Hiyerarşik Kümeleme Dendongramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average, leaf_font_size=10)
plt.show(block=True)


# 10 küme sınırı girilmesi.
plt.figure(figsize=(19,5))
plt.title("Hiyerarşik Kümeleme Dendongramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show(block=True)

################################
# Kume Sayısını Belirlemek
################################

plt.figure(figsize=(7,5))
dend = dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=0.6, color="r", linestyle="--")
plt.show(block=True)


# Birden fazla çizgi koymak için
plt.figure(figsize=(7,5))
dend = dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=0.5, color="b", linestyle="--")
plt.axhline(y=0.6, color="r", linestyle="--")
plt.show(block=True)

################################
# Final Modeli Oluşturmak
################################

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, linkage="average")

clusters = cluster.fit_predict(df)

df = pd.read_csv("DS_Bootcamp_13/datasets/USArrests.csv", index_col=0)

df["hi_cluster_no"] = clusters
df.head()
df["hi_cluster_no"] = df["hi_cluster_no"] + 1
df.head()

df["kmeans_cluster_no"] = clusters_kmeans
df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1
df.head()

len(df[df["hi_cluster_no"] == 1])
len(df[df["kmeans_cluster_no"] == 1])
len(df[(df["hi_cluster_no"] == 1) & (df["kmeans_cluster_no"] == 1)])

len(df[df["hi_cluster_no"] == 2])
len(df[df["kmeans_cluster_no"] == 2])
len(df[(df["hi_cluster_no"] == 2) & (df["kmeans_cluster_no"] == 2)])

len(df[df["hi_cluster_no"] == 3])
len(df[df["kmeans_cluster_no"] == 3])
len(df[(df["hi_cluster_no"] == 3) & (df["kmeans_cluster_no"] == 3)])

len(df[df["hi_cluster_no"] == 4])
len(df[df["kmeans_cluster_no"] == 4])
len(df[(df["hi_cluster_no"] == 4) & (df["kmeans_cluster_no"] == 4)])

len(df[df["hi_cluster_no"] == 5])
len(df[df["kmeans_cluster_no"] == 5])
len(df[(df["hi_cluster_no"] == 5) & (df["kmeans_cluster_no"] == 5)])


################################
# Principal Component Analysis
################################

df = pd.read_csv("DS_Bootcamp_13/datasets/hitters.csv")
df.head()

num_cols = [col for col in df.columns if (str(df[col].dtypes) not in ["category", "object", "bool"]) & ("Salary" not in col)]

len(num_cols)

df[num_cols].head()

df = df[num_cols]
df.isnull().sum()

df = StandardScaler().fit_transform(df)
type(df)

pca = PCA()
pca_fit = pca.fit_transform(df)
type(pca_fit)

# Değişkenlerin açıkladığı bilgi oranları
pca.explained_variance_ratio_
type(pca.explained_variance_ratio_)

# kümülatif bilgi oranları (1 bileşen ile yüzde, 2 bileşen ile yüzde...)
np.cumsum(pca.explained_variance_ratio_)

################################
# Optimum Bileşen Sayısı
################################

pca = PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Bileşen Sayısı")
plt.ylabel("Kümülatif Varyans Oranı")
plt.show(block=True)

################################
# Final PCA'in Oluşturulması
################################

pca = PCA(n_components=3)
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)

################################
# BONUS: Principal Component Regression
################################

# Burda pca ile 3'e indirilien sayısal değişkenlere kategorikleri encode yapıp ekliyoruz.

df = pd.read_csv("DS_Bootcamp_13/datasets/hitters.csv")
df.head()
df.shape
len(pca_fit)

num_cols = [col for col in df.columns if (str(df[col].dtypes) not in ["category", "object", "bool"]) & ("Salary" not in col)]
len(num_cols)

others = [col for col in df.columns if col not in num_cols]
len(others)

pd.DataFrame(pca_fit, columns=["PC1", "PC2", "PC3"])

df[others]

final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1", "PC1", "PC3"]), df[others]], axis=1)

final_df.head()

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Kategorik değişkenlerin 2 değerli olduğunu bildiğimiz için label encoder kullandık
# one_hot_encoder da kullanabilirdik direk (get_dummies'li)
def label_encoder(dataframe, binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe

for col in ["NewLeague", "Division", "League"]:
    label_encoder(final_df, col)

final_df.head()

final_df.isnull().sum()
final_df.dropna(inplace=True)
final_df.isnull().sum()

y = final_df["Salary"]
X = final_df.drop(["Salary"], axis=1)

lm = LinearRegression()
rmse = np.mean(np.sqrt(-cross_val_score(lm,
                                        X,
                                        y,
                                        cv=5,
                                        scoring="neg_mean_squared_error")))

rmse
y.mean()

cart = DecisionTreeRegressor()
rmse = np.mean(np.sqrt(-cross_val_score(cart,
                                        X,
                                        y,
                                        cv=5,
                                        scoring="neg_mean_squared_error")))

rmse

cart_params = {'max_depth': range(1, 11),
               "min_samples_split": range(2, 20)}

# GridSearchCV
cart_best_grid = GridSearchCV(cart,
                              cart_params,
                              cv=5,
                              n_jobs=-1,
                              verbose=True).fit(X, y)

cart_best_grid.best_params_

cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(cart_final,
                                        X,
                                        y,
                                        cv=5,
                                        scoring="neg_mean_squared_error")))

rmse

################################
# BONUS: PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme
################################

################################
# Breast Cancer
################################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("DS_Bootcamp_13/datasets/breast_cancer.csv")
df.head()

y = df["diagnosis"]
X = df.drop(["diagnosis", "id"], axis=1)

# 2 boyutlu pca_df yapıyor.
def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df

pca_df = create_pca_df(X, y)
pca_df.head()
pca_df.shape

def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

plot_pca(pca_df, "diagnosis")


################################
# Iris
################################

import seaborn as sns
df = sns.load_dataset("iris")

y = df["species"]
X = df.drop(["species"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "species")


################################
# Diabetes
################################

df = pd.read_csv("DS_Bootcamp_13/datasets/diabetes.csv")

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

pca_df = create_pca_df(X, y)

plot_pca(pca_df, "Outcome")


################################################
# End-to-End Diabetes Machine Learning Pipeline I
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Base Models
# 4. Automated Hyperparameter Optimization
# 5. Stacking & Ensemble Learning
# 6. Prediction for a New Observation
# 7. Pipeline Main Function

import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: "%.3f" % x)

################################################
# 1. Exploratory Data Analysis
################################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(dataframe, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(dataframe[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Değişkenleri ayırır.
    Parameters
    ----------
    dataframe : dataframe
        Veriseti
    cat_th : int, float
        Kategorik Değişken Tresshold
    car_th : int, float
        Kardinal Değişken Tresshold

    Returns
        cat_cols
        num_cols
        cat_but_car

    Notes
        cat_cols + num_cols + car_but_car = toplam değişken sayısı
    -------

    """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if (dataframe[col].dtypes in ["int", "float"]) & (dataframe[col].nunique() < cat_th)]
    cat_but_car = [col for col in dataframe.columns if (str(dataframe[col].dtypes) in ["category", "object"]) & (dataframe[col].nunique() > car_th)]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables {dataframe.shape[1]}")
    print(f"cat_cols {len(cat_cols)}")
    print(f"num_cols {len(num_cols)}")
    print(f"cat_but_car {len(cat_but_car)}")
    print(f"num_but_cat {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


df = pd.read_csv("DS_Bootcamp_13/datasets/diabetes.csv")
df.head()

check_df(df)


# Değişken türlerinin ayrıştırılması
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5 , car_th=20)

# Kategorik değişkenlerin incelenmesi
for col in cat_cols:
    cat_summary(df, col)

# Sayısal değişkenlerin incelenmesi
df[num_cols].describe().T

for col in num_cols:
    num_summary(df, col, plot=True)

# Sayısal değişkenkerin birbirleri ile korelasyonu
correlation_matrix(df, num_cols)

# Target ile sayısal değişkenlerin incelemesi
for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

# bu veri setinde hedef değişken dışında kategorik değişken olmadığı için
# target_summary_with_cat fonksiyonunu şimdilik kullanmadık.

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def one_hot_encoder(dataframe, categorical_cols, drop_first=True, dtype=int, dummy_na=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=dtype, dummy_na=dummy_na)
    return dataframe

df.head()

# Değişken isimleri büyütmek
df.columns = [col.upper() for col in df.columns]

# Glucose
df['NEW_GLUCOSE_CAT'] = pd.cut(x=df['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])

# Age
df.loc[(df['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
df.loc[(df['AGE'] >= 35) & (df['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
df.loc[(df['AGE'] > 55), "NEW_AGE_CAT"] = 'old'


# BMI
df['NEW_BMI_RANGE'] = pd.cut(x=df['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                             labels=["underweight", "healty", "overweight", "obese"])

# BloodPressure
df['NEW_BLOODPRESSURE'] = pd.cut(x=df['BLOODPRESSURE'], bins=[-1, 79, 89, 123], labels=["normal", "hs1", "hs2"])

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

for col in cat_cols:
    cat_summary(df, col)

for col in cat_cols:
    target_summary_with_cat(df, "OUTCOME", col)

cat_cols = [col for col in cat_cols if "OUTCOME" not in col]

df = one_hot_encoder(df, cat_cols, drop_first=True)

check_df(df)

df.columns = [col.upper() for col in df.columns]

# Son güncel değişken türlerimi tutuyorum.
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
cat_cols = [col for col in cat_cols if "OUTCOME" not in col]

for col in num_cols:
    print(col, check_outlier(df, col, 0.05, 0.95))

replace_with_thresholds(df, "INSULIN")


# Standartlaştırma
X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

check_df(X)

def diabetes_data_prep(dataframe):
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # Glucose
    dataframe['NEW_GLUCOSE_CAT'] = pd.cut(x=dataframe['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])

    # Age
    dataframe.loc[(dataframe['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 35) & (dataframe['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
    dataframe.loc[(dataframe['AGE'] > 55), "NEW_AGE_CAT"] = 'old'

    # BMI
    dataframe['NEW_BMI_RANGE'] = pd.cut(x=dataframe['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                                        labels=["underweight", "healty", "overweight", "obese"])

    # BloodPressure
    dataframe['NEW_BLOODPRESSURE'] = pd.cut(x=dataframe['BLOODPRESSURE'], bins=[-1, 79, 89, 123],
                                            labels=["normal", "hs1", "hs2"])

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)

    cat_cols = [col for col in cat_cols if "OUTCOME" not in col]

    df = one_hot_encoder(dataframe, cat_cols, drop_first=True)

    df.columns = [col.upper() for col in df.columns]

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

    cat_cols = [col for col in cat_cols if "OUTCOME" not in col]

    replace_with_thresholds(df, "INSULIN")

    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    y = df["OUTCOME"]
    X = df.drop(["OUTCOME"], axis=1)

    return X, y

df = pd.read_csv("DS_Bootcamp_13/datasets/diabetes.csv")

check_df(df)

X, y = diabetes_data_prep(df)

check_df(X)

######################################################
# 3. Base Models
######################################################

import warnings
# Bu belirli uyarıyı kapatmak için
warnings.filterwarnings("ignore", message=".*c_contiguous.*")

# UserWarning türündeki uyarıları kapat
warnings.filterwarnings("ignore", category=UserWarning)

# Uyarıyı tamamen kapatmak için (tavsiye edilmez, çünkü diğer önemli uyarıları da kapatabilir)
warnings.filterwarnings("ignore")

# Uyarıyı tamamen açmak için
warnings.filterwarnings("default")


def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(verbose=-1)),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

base_models(X, y, scoring="accuracy")

#accuracy: 0.7604 (LR)
#accuracy: 0.7617 (KNN)
#accuracy: 0.7656 (SVC)
#accuracy: 0.6862 (CART)
#accuracy: 0.7695 (RF)
#accuracy: 0.7578 (Adaboost)
#accuracy: 0.7461 (GBM)
#accuracy: 0.7526 (XGBoost)
#accuracy: 0.7383 (LightGBM)
#accuracy: 0.7669 (CatBoost)

######################################################
# 4. Automated Hyperparameter Optimization
######################################################

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(verbose=-1), lightgbm_params),
               ('CatBoost', CatBoostClassifier(verbose=False), catboost_params)
               ]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)
type(best_models)
best_models

######################################################
# 5. Stacking & Ensemble Learning
######################################################

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"]),
                                              ('CatBoost', best_models["CatBoost"])
                                              ],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)

# Accuracy: 0.7682291666666666
# F1Score: 0.6357131545093341
# ROC_AUC: 0.8370219640738977

# en iyi modelleri ya da farklı çalışma prensibi olan modelleri seçip onlar üzerinden tahmin oluşturuyoruz.

######################################################
# 6. Prediction for a New Observation
######################################################

X.columns
random_user = X.sample(1, random_state=45)
voting_clf.predict(random_user)

joblib.dump(voting_clf, "voting_clf.pkl")

new_model = joblib.load("voting_clf.pkl")
new_model.predict(random_user)

################################################
# End-to-End Diabetes Machine Learning Pipeline II
################################################

# pip install catboost

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

################################################
# Helper Functions
################################################
# from helpers import grab_col_names, outlier_thresholds, replace_with_thresholds, one_hot_encoder, diabetes_data_prep, base_models.......

# utils.py
# helpers.py

# Data Preprocessing & Feature Engineering
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Değişkenleri ayırır.
    Parameters
    ----------
    dataframe : dataframe
        Veriseti
    cat_th : int, float
        Kategorik Değişken Tresshold
    car_th : int, float
        Kardinal Değişken Tresshold

    Returns
        cat_cols
        num_cols
        cat_but_car

    Notes
        cat_cols + num_cols + car_but_car = toplam değişken sayısı
    -------

    """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [col for col in dataframe.columns if (dataframe[col].dtypes in ["int", "float"]) & (dataframe[col].nunique() < cat_th)]
    cat_but_car = [col for col in dataframe.columns if (str(dataframe[col].dtypes) in ["category", "object"]) & (dataframe[col].nunique() > car_th)]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables {dataframe.shape[1]}")
    print(f"cat_cols {len(cat_cols)}")
    print(f"num_cols {len(num_cols)}")
    print(f"cat_but_car {len(cat_but_car)}")
    print(f"num_but_cat {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def diabetes_data_prep(dataframe):
    dataframe.columns = [col.upper() for col in dataframe.columns]

    # Glucose
    dataframe['NEW_GLUCOSE_CAT'] = pd.cut(x=dataframe['GLUCOSE'], bins=[-1, 139, 200], labels=["normal", "prediabetes"])

    # Age
    dataframe.loc[(dataframe['AGE'] < 35), "NEW_AGE_CAT"] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 35) & (dataframe['AGE'] <= 55), "NEW_AGE_CAT"] = 'middleage'
    dataframe.loc[(dataframe['AGE'] > 55), "NEW_AGE_CAT"] = 'old'

    # BMI
    dataframe['NEW_BMI_RANGE'] = pd.cut(x=dataframe['BMI'], bins=[-1, 18.5, 24.9, 29.9, 100],
                                        labels=["underweight", "healty", "overweight", "obese"])

    # BloodPressure
    dataframe['NEW_BLOODPRESSURE'] = pd.cut(x=dataframe['BLOODPRESSURE'], bins=[-1, 79, 89, 123],
                                            labels=["normal", "hs1", "hs2"])

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, cat_th=5, car_th=20)

    cat_cols = [col for col in cat_cols if "OUTCOME" not in col]

    df = one_hot_encoder(dataframe, cat_cols, drop_first=True)

    cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

    replace_with_thresholds(df, "INSULIN")

    X_scaled = StandardScaler().fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns)

    y = df["OUTCOME"]
    X = df.drop(["OUTCOME"], axis=1)

    return X, y

# Base Models
def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier(verbose=-1)),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

# Hyperparameter Optimization

# config.py

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}


catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(verbose=-1), lightgbm_params),
               ('CatBoost', CatBoostClassifier(verbose=False), catboost_params)
               ]

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

# Stacking & Ensemble Learning
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"]),
                                              #('CatBoost', best_models["CatBoost"])
                                              ],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

################################################
# Pipeline Main Function
################################################

import warnings
# Bu belirli uyarıyı kapatmak için
warnings.filterwarnings("ignore", message=".*c_contiguous.*")

# Uyarıyı tamamen kapatmak için (tavsiye edilmez, çünkü diğer önemli uyarıları da kapatabilir)
warnings.filterwarnings("ignore")


def main():
    df = pd.read_csv("/Users/acibal/DataScience/Miuul/DS_Bootcamp_13/datasets/diabetes.csv") #full path
    X, y = diabetes_data_prep(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    voting_clf = voting_classifier(best_models, X, y)
    joblib.dump(voting_clf, "voting_clf.pkl")
    return voting_clf

# terminalden çalışır.
if __name__ == "__main__":
    print("İşlem başladı")
    main()

# git github
# makefile
# veri tabanlarından
# log
# class
# docker
# requirement.txt









from functools import reduce
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rc,rcParams
import datetime as dt
import math
import scipy.stats as st
import random
import joblib
import itertools
import missingno as msno
import graphviz
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from skompiler import skompile
from statsmodels.stats.proportion import proportions_ztest
from sklearn.impute import KNNImputer
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.multicomp import MultiComparison
from mlxtend.frequent_patterns import apriori, association_rules
# from surprise import Reader, SVD, Dataset, accuracy
# from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split, RandomizedSearchCV, validation_curve
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, export_text
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve, RocCurveDisplay, mean_squared_error, mean_absolute_error
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage, dendrogram
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier







##################################################
# Introduction to Text Mining and Natural Language Processing
##################################################

##################################################
# Sentiment Analysis and Sentiment Modeling for Amazon Reviews
##################################################

# 1. Text Preprocessing
# 2. Text Visualization
# 3. Sentiment Analysis
# 4. Feature Engineering
# 5. Sentiment Modeling

# !pip install nltk
# !pip install textblob
# !pip install wordcloud

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

##################################################
# 1. Text Preprocessing
##################################################

df = pd.read_csv("NLP/datasets/amazon_reviews.csv", sep=",")
df.head()

###############################
# Normalizing Case Folding
###############################

df['reviewText'] = df['reviewText'].str.lower()

###############################
# Punctuations
###############################

df['reviewText'] = df['reviewText'].str.replace(r'[^\w\s]', ' ', regex=True)

# regular expression

###############################
# Numbers
###############################

df['reviewText'] = df['reviewText'].str.replace(r'\d', ' ', regex=True)

###############################
# Stopwords
###############################
# import nltk
# nltk.download('stopwords')

sw = stopwords.words('english')
type(sw)

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

###############################
# Rarewords
###############################

temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()

drops = temp_df[temp_df <= 1]

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

###############################
# Tokenization
###############################

# nltk.download("punkt")

df["reviewText"].apply(lambda x: TextBlob(x).words)
# atamadık kısa yolu görmek için yaptık.

###############################
# Lemmatization
###############################

# nltk.download('wordnet')
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

# Stemming de köklerine ayırma işlemidir. Lemmatization tercih edilir.

##################################################
# 2. Text Visualization
##################################################

###############################
# Terim Frekanslarının Hesaplanması
###############################

tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

###############################
# Barplot
###############################

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show(block=True)

###############################
# Wordcloud
###############################

text = " ".join(i for i in df.reviewText)
type(text)
len(text)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show(block=True)


wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show(block=True)

#wordcloud.to_file("wordcloud.png")

###############################
# Şablonlara Göre Wordcloud
###############################

tr_mask = np.array(Image.open("NLP/tr.png"))
type(tr_mask)

wc = WordCloud(background_color="white",
               max_words=1000,
               mask=tr_mask,
               contour_width=3,
               contour_color="firebrick").generate(text)

plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show(block=True)

##################################################
# 3. Sentiment Analysis
##################################################

df["reviewText"].head()

# nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
sia.polarity_scores("The film was awesome")


sia.polarity_scores("I liked this music but it is not good as the other one")
type(sia.polarity_scores("I liked this music but it is not good as the other one"))

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x))

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["polarity_score"] = df["reviewText"].apply(lambda x: sia.polarity_scores(x)["compound"])

###############################
# 4. Feature Engineering
###############################

df["reviewText"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"].value_counts()

df.groupby("sentiment_label")["overall"].mean()

df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

y = df["sentiment_label"]
X = df["reviewText"]

###############################
# Count Vectors
###############################

# Count Vectors: frekans temsiller
# TF-IDF Vectors: normalize edilmiş frekans temsiller
# Word Embeddings (Word2Vec, GloVe, BERT vs)


# words
# kelimelerin nümerik temsilleri

# characters
# karakterlerin nümerik temsilleri

# ngram
a = """Bu örneği anlaşılabilmesi için daha uzun bir metin üzerinden göstereceğim.
N-gram'lar birlikte kullanılan kelimelerin kombinasyolarını gösterir ve feature üretmek için kullanılır"""

TextBlob(a).ngrams(3)

###############################
# Count Vectors
###############################

from sklearn.feature_extraction.text import CountVectorizer

corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']
type(corpus)

# word frekans
vectorizer = CountVectorizer()
X_c = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out()
X_c.toarray()
type(X_c)

# n-gram frekans
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X_n = vectorizer2.fit_transform(corpus)
vectorizer2.get_feature_names_out()
X_n.toarray()


vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(X)

vectorizer.get_feature_names_out()[10:15]
X_count.toarray()[10:15]


###############################
# TF-IDF
###############################

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)


tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(X)


###############################
# 5. Sentiment Modeling
###############################

# 1. Text Preprocessing
# 2. Text Visualization
# 3. Sentiment Analysis
# 4. Feature Engineering
# 5. Sentiment Modeling

###############################
# Logistic Regression
###############################

log_model = LogisticRegression().fit(X_tf_idf_word, y)

cross_val_score(log_model,
                X_tf_idf_word,
                y,
                scoring="accuracy",
                cv=5).mean()

new_review = pd.Series("this product is great")
new_review = pd.Series("look at that shit very bad")
new_review = pd.Series("it was good but I am sure that it fits me")
type(new_review)

new_review = TfidfVectorizer().fit(X).transform(new_review)

log_model.predict(new_review)

random_review = pd.Series(df["reviewText"].sample(1).values)
type(random_review)

new_review = TfidfVectorizer().fit(X).transform(random_review)

log_model.predict(new_review)

###############################
# Random Forests
###############################

# Count Vectors
rf_model = RandomForestClassifier().fit(X_count, y)
cross_val_score(rf_model, X_count, y, cv=5, n_jobs=-1).mean()

# TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)
cross_val_score(rf_model, X_tf_idf_word, y, cv=5, n_jobs=-1).mean()

# TF-IDF N-GRAM
rf_model = RandomForestClassifier().fit(X_tf_idf_ngram, y)
cross_val_score(rf_model, X_tf_idf_ngram, y, cv=5, n_jobs=-1).mean()

###############################
# Hiperparametre Optimizasyonu
###############################

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [8, None],
             "max_features": [7, "auto"],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [100, 200]}

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=1).fit(X_count, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_count, y)

cross_val_score(rf_final, X_count, y, cv=5, n_jobs=-1).mean()






##################################################
# TIME SERIES
##################################################

##################################################
# Smoothing Methods (Holt-Winters)
##################################################

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt

warnings.filterwarnings('ignore')

############################
# Veri Seti
############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

data = sm.datasets.co2.load_pandas()
y = data.data

y = y['co2'].resample('MS').mean()

y.isnull().sum()

y = y.fillna(y.bfill())

# ffill() önceki değer bfill() sonraki değer

y.plot(figsize=(15, 6))
plt.show()

############################
# Holdout
############################

train = y[:'1997-12-01']
len(train)  # 478 ay

# 1998'ilk ayından 2001'in sonuna kadar test set.
test = y['1998-01-01':]
len(test)  # 48 ay

##################################################
# Zaman Serisi Yapısal Analizi
##################################################

# Durağanlık Testi (Dickey-Fuller Testi)

def is_stationary(y):

    # "HO: Non-stationary"
    # "H1: Stationary"

    p_value = sm.tsa.stattools.adfuller(y)[1]
    if p_value < 0.05:
        print(F"Result: Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")
    else:
        print(F"Result: Non-Stationary (H0: non-stationary, p-value: {round(p_value, 3)})")

is_stationary(y)

# Zaman Serisi Bileşenleri ve Durağanlık Testi
def ts_decompose(y, model="additive", stationary=False):
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g', label='Seasonality & Mean: ' + str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r', label='Residuals & Mean: ' + str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show(block=True)

    if stationary:
        is_stationary(y)

ts_decompose(y, stationary=True)

##################################################
# Single Exponential Smoothing
##################################################

# SES = Level

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.5)

y_pred = ses_model.forecast(48)
# SES'de predict değil forecest var.
# Test testinde 48 ay olduğu için burada da 48 ya tahmin et parametresini giriyoruz.

mean_absolute_error(test, y_pred)
# mae yerine (mse) mean squared error ya da (rmse) root mean squared error da kullanılabilir.

train.plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show(block=True)


train["1985":].plot(title="Single Exponential Smoothing")
test.plot()
y_pred.plot()
plt.show(block=True)

def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show(block=True)

# bu fonksiyondaki 1985 daha yakından tahminleri görmek için yazdık. Genellenemez bir girdi. Veri setine göre değiştir ya da kaldır.

plot_co2(train, test, y_pred, "Single Exponential Smoothing")

ses_model.params

############################
# Hyperparameter Optimization
############################

def ses_optimizer(train, alphas, step=48):

    best_alpha, best_mae = None, float("inf")

    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)

        if mae < best_mae:
            best_alpha, best_mae = alpha, mae

        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_mae

alphas = np.arange(0.5, 1, 0.01)
# 0.1'den 1'e kadar da denenebilir. Ama biliyoruz ki alfanın 0.5'ten büyük olması daha mantıklı.

# yt_sapka = a * yt-1 + (1-a)* (yt_-1)_sapka

ses_optimizer(train, alphas)

best_alpha, best_mae = ses_optimizer(train, alphas)

############################
# Final SES Model
############################

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha)
y_pred = ses_model.forecast(48)

plot_co2(train, test, y_pred, "Single Exponential Smoothing")


##################################################
# Double Exponential Smoothing (DES)
##################################################

# DES: Level (SES) + Trend
# toplamsal ve çarpımsal modeller:
# mevsimsellik ve artık bileşenleri trendten bağımsızsa toplamsal seri değilse çarpımsal seri
# grafikten mevsimsellik ve artıklar sıfır etrafında dağılıyorsa, trendten bağımsızdır ve toplamsaldır.
# artıklar = error'lar
# eğer grafiklerden yorumlamak istemezsek iki modeli de kurup düşük hatalı modeli seçiyoruz.
# y(t) = Level + Trend + Seasonality + Noise
# y(t) = Level * Trend * Seasonality * Noise


ts_decompose(y) # burdan grafikleriden görebiliriz.

des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.5,
                                                         smoothing_trend=0.5)

y_pred = des_model.forecast(48)

plot_co2(train, test, y_pred, "Double Exponential Smoothing")

############################
# Hyperparameter Optimization
############################

def des_optimizer(train, alphas, betas, step=48):
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha, smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_mae

alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)

best_alpha, best_beta, best_mae = des_optimizer(train, alphas, betas)

############################
# Final DES Model
############################

final_des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=best_alpha,
                                                               smoothing_slope=best_beta)
# trend="add" additive mevsimsellik ve artıklar trendten bağımsız olduğu için. (toplamsal seri)
# trend="mul" multiplicative
# final_des_model = ExponentialSmoothing(train, trend="mul").fit(smoothing_level=best_alpha,
#                                                                smoothing_slope=best_beta)
# iki türlü de hataya bakılabilir eğer görsel yorumuna güvenmiyorsan.

y_pred = final_des_model.forecast(48)

plot_co2(train, test, y_pred, "Double Exponential Smoothing")

##################################################
# Triple Exponential Smoothing (Holt-Winters)
##################################################

# TES = SES + DES + Mevsimsellik

tes_model = ExponentialSmoothing(train,
                                 trend="add",
                                 seasonal="add",
                                 seasonal_periods=12).fit(smoothing_level=0.5,
                                                          smoothing_slope=0.5,
                                                          smoothing_seasonal=0.5)
# burda seasonal_periods'un 12 olduğunu görselden anlıyoruz.
# yani mevsimsellik yıllık periodlarla tekrar ediyor.

y_pred = tes_model.forecast(48)
plot_co2(train, test, y_pred, "Triple Exponential Smoothing")

############################
# Hyperparameter Optimization
############################

alphas = betas = gammas = np.arange(0.10, 1, 0.10)

abg = list(itertools.product(alphas, betas, gammas))

def tes_optimizer(train, abg, step=48):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg)


############################
# Final TES Model
############################

final_tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)

y_pred = final_tes_model.forecast(48)

plot_co2(train, test, y_pred, "Triple Exponential Smoothing")

##################################################
# Statistical Methods
##################################################

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings('ignore')

############################
# Veri Seti
############################

# Atmospheric CO2 from Continuous Air Samples at Mauna Loa Observatory, Hawaii, U.S.A.
# Period of Record: March 1958 - December 2001

data = sm.datasets.co2.load_pandas()
y = data.data
y = y['co2'].resample('MS').mean()
y = y.fillna(y.bfill())
train = y[:'1997-12-01']
test = y['1998-01-01':]

##################################################
# ARIMA(p, d, q): (Autoregressive Integrated Moving Average)
##################################################

arima_model = ARIMA(train, order=(1, 1, 1)).fit()

arima_model.summary()
# modelin istatistiki çıktısı. (bilgi amaçlı)

y_pred = arima_model.get_forecast(48).predicted_mean

# ÖNCESİ KODLAR
# y_pred = arima_model.forecast(48)[0]
# y_pred = pd.Series(y_pred, index=test.index)

def plot_co2(train, test, y_pred, title):
    mae = mean_absolute_error(test, y_pred)
    train["1985":].plot(legend=True, label="TRAIN", title=f"{title}, MAE: {round(mae,2)}")
    test.plot(legend=True, label="TEST", figsize=(6, 4))
    y_pred.plot(legend=True, label="PREDICTION")
    plt.show(block=True)

plot_co2(train, test, y_pred, "ARIMA")
# bi sıkıntı var gibi trendi yakalayamıyor

############################
# Hyperparameter Optimization (Model Derecelerini Belirleme)
############################

############################
# AIC & BIC İstatistiklerine Göre Model Derecesini Belirleme
############################

p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))


def arima_optimizer_aic(train, orders):
    best_aic, best_params = float("inf"), None
    for order in orders:
        try:
            arima_model_result = ARIMA(train, order=order).fit()
            aic = arima_model_result.aic
            if aic < best_aic:
                best_aic, best_params = aic, order
            print('ARIMA%s AIC=%.2f' % (order, aic))
        except:
            continue
    print('Best ARIMA%s AIC=%.2f' % (best_params, best_aic))
    return best_params

best_params_aic = arima_optimizer_aic(train, pdq)

############################
# Final Model
############################

arima_model = ARIMA(train, order=best_params_aic).fit()
y_pred = arima_model.forecast(steps=len(test))

plot_co2(train, test, y_pred, "ARIMA")


##################################################
# SARIMA(p, d, q): (Seasonal Autoregressive Integrated Moving-Average)
##################################################

model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 12))

sarima_model = model.fit()

y_pred_test = sarima_model.get_forecast(steps=48)

y_pred = y_pred_test.predicted_mean

# y_pred = pd.Series(y_pred, index=test.index) artık gereksiz

plot_co2(train, test, y_pred, "SARIMA")


############################
# Hyperparameter Optimization (Model Derecelerini Belirleme)
############################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), None, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit()
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, aic))
            except:
                continue
    print('SARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_aic(train, pdq, seasonal_pdq)


############################
# Final Model
############################

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit()

y_pred_test = sarima_final_model.get_forecast(steps=48)

y_pred = y_pred_test.predicted_mean
# y_pred = pd.Series(y_pred, index=test.index) gereksiz

plot_co2(train, test, y_pred, "SARIMA")

##################################################
# BONUS: MAE'ye Göre SARIMA Optimizasyonu
##################################################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


def sarima_optimizer_mae(train, pdq, seasonal_pdq):
    best_mae, best_order, best_seasonal_order = float("inf"), None, None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                sarima_model = model.fit()
                y_pred_test = sarima_model.get_forecast(steps=48)
                y_pred = y_pred_test.predicted_mean
                mae = mean_absolute_error(test, y_pred)
                if mae < best_mae:
                    best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
                print('SARIMA{}x{}12 - MAE:{}'.format(param, param_seasonal, mae))
            except:
                continue
    print('SARIMA{}x{}12 - MAE:{}'.format(best_order, best_seasonal_order, best_mae))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_mae(train, pdq, seasonal_pdq)

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit()

y_pred_test = sarima_final_model.get_forecast(steps=48)
y_pred = y_pred_test.predicted_mean
# y_pred = pd.Series(y_pred, index=test.index) gereksiz

plot_co2(train, test, y_pred, "SARIMA")


############################
# Final Model
############################

# tüm veriye göre model (train değil y)
model = SARIMAX(y, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit()

# gelecek 6 ay ile ilgili tahminler
feature_predict = sarima_final_model.get_forecast(steps=6)
feature_predict = feature_predict.predicted_mean

# tahnminlerin hatasını da zamanla verileri kıyaslayarak yapabiliriz. 6 ay sonra
# 6 ay sonra tekrar düzeltmeleri yaptırıp yeni bir model kurabiliriz.


#################################
# Airline Passenger Forecasting
#################################

import itertools
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')

#################################
# Verinin Görselleştirilmesi
#################################

df = pd.read_csv('Time_Series/datasets/airline-passengers.csv', index_col='month', parse_dates=True)
# indexlerde tarihler olsun diye index_col="month" girildi.
# tarihler algılansın diye parse_dates=True girildi.

df.shape
df.head()

df[['total_passengers']].plot(title='Passengers Data')
plt.show(block=True)

df.index

df.index.freq = "MS"
# tarihler aylık ama bunla sadece tarihlerin aylık olduğu bilgisini verdik.
df.index


train = df[:120]
test = df[120:]

len(test)
# tahmin ayımız 24 ay

#################################
# Single Exponential Smoothing
#################################

def ses_optimizer(train, alphas, step=48):
    best_alpha, best_mae = None, float("inf")
    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_mae = alpha, mae
        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_mae

alphas = np.arange(0.01, 1, 0.10)
best_alpha, best_mae = ses_optimizer(train, alphas, step=24)
# best_alpha: 0.11 best_mae: 82.528

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=best_alpha)
y_pred = ses_model.forecast(24)

def plot_prediction(y_pred, label):
    train["total_passengers"].plot(legend=True, label="TRAIN")
    test["total_passengers"].plot(legend=True, label="TEST")
    y_pred.plot(legend=True, label="PREDICTION")
    plt.title("Train, Test and Predicted Test Using "+label)
    plt.show(block=True)

plot_prediction(y_pred, "Single Exponential Smoothing")


#################################
# Double Exponential Smoothing
#################################

def des_optimizer(train, alphas, betas, step=48):
    best_alpha, best_beta, best_mae = None, None, float("inf")
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=alpha, smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            if mae < best_mae:
                best_alpha, best_beta, best_mae = alpha, beta, mae
            print("alpha:", round(alpha, 2), "beta:", round(beta, 2), "mae:", round(mae, 4))
    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_mae:", round(best_mae, 4))
    return best_alpha, best_beta, best_mae

alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)

best_alpha, best_beta, best_mae = des_optimizer(train, alphas, betas, step=24)
# best_alpha: 0.01 best_beta: 0.11 best_mae: 54.1036

des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=best_alpha,
                                                         smoothing_slope=best_beta)
y_pred = des_model.forecast(24)

plot_prediction(y_pred, "Double Exponential Smoothing")

#################################
# Triple Exponential Smoothing (Holt-Winters)
#################################

def tes_optimizer(train, abg, step=48):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    return best_alpha, best_beta, best_gamma, best_mae

alphas = betas = gammas = np.arange(0.10, 1, 0.20)
abg = list(itertools.product(alphas, betas, gammas))

best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, abg, step=24)

# best_alpha: 0.3 best_beta: 0.3 best_gamma: 0.5 best_mae: 11.9947

# ÖNCE ADD İLE
tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_slope=best_beta, smoothing_seasonal=best_gamma)

y_pred = tes_model.forecast(24)

plot_prediction(y_pred, "Triple Exponential Smoothing ADD")


# SONRA MUL İLE
tes_model = ExponentialSmoothing(train, trend="mul", seasonal="mul", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_slope=best_beta, smoothing_seasonal=best_gamma)

y_pred = tes_model.forecast(24)

plot_prediction(y_pred, "Triple Exponential Smoothing MUL")


# Görsel kıyaslanıp hangisi daha iyi seçilir.
# Nümerik olarak seçmek için tes_optimizer fonnksiyonunun içindeki trend="add" ve seasonal="add" yerine "mul" yazıp sonuçları gözlemleyebiliriz.

##################################################
# ARIMA(p, d, q): (Autoregressive Integrated Moving Average)
##################################################

p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))


def arima_optimizer_aic(train, orders):
    best_aic, best_params = float("inf"), None
    for order in orders:
        try:
            arma_model_result = ARIMA(train, order).fit()
            aic = arma_model_result.aic
            if aic < best_aic:
                best_aic, best_params = aic, order
            print('ARIMA%s AIC=%.2f' % (order, aic))
        except:
            continue
    print('Best ARIMA%s AIC=%.2f' % (best_params, best_aic))
    return best_params

best_params_aic = arima_optimizer_aic(train, pdq)

arima_model = ARIMA(train, best_params_aic).fit()
y_pred = arima_model.forecast(24)
mean_absolute_error(test, y_pred)
# 51.1806294123169
# bendeki sonuç 206
# ARIMA'DA GENE HATA VAR TRENDİ YAKALAYAMADI
plot_prediction(pd.Series(y_pred, index=test.index), "ARIMA")


##################################################
# SARIMA
##################################################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                results = sarimax_model.fit()
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param, param_seasonal
                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, aic))
            except:
                continue
    print('SARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order, best_aic))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_aic(train, pdq, seasonal_pdq)

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)
y_pred_test = sarima_final_model.get_forecast(steps=24)

y_pred = y_pred_test.predicted_mean
mean_absolute_error(test, y_pred)
# 68.57726545235921

plot_prediction(pd.Series(y_pred, index=test.index), "SARIMA")


# MAE


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

def sarima_optimizer_mae(train, pdq, seasonal_pdq):
    best_mae, best_order, best_seasonal_order = float("inf"), float("inf"), None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                sarima_model = model.fit()
                y_pred_test = sarima_model.get_forecast(steps=24)
                y_pred = y_pred_test.predicted_mean
                mae = mean_absolute_error(test, y_pred)

                # mae = fit_model_sarima(train, val, param, param_seasonal)

                if mae < best_mae:
                    best_mae, best_order, best_seasonal_order = mae, param, param_seasonal
                print('SARIMA{}x{}12 - MAE:{}'.format(param, param_seasonal, mae))
            except:
                continue
    print('SARIMA{}x{}12 - MAE:{}'.format(best_order, best_seasonal_order, best_mae))
    return best_order, best_seasonal_order

best_order, best_seasonal_order = sarima_optimizer_mae(train, pdq, seasonal_pdq)

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)
y_pred_test = sarima_final_model.get_forecast(steps=24)
y_pred = y_pred_test.predicted_mean
mean_absolute_error(test, y_pred)
# 30.623362595882828

plot_prediction(pd.Series(y_pred, index=test.index), "SARIMA")



# Final model
# en iyi sonuç en az hata tes ile aldığımız için final modelimizi onla kuruyoruz.

tes_model_final = ExponentialSmoothing(df, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_slope=best_beta, smoothing_seasonal=best_gamma)

tes_model_final.forecast(6)


#####################################################
# Demand Forecasting
#####################################################

# Store Item Demand Forecasting Challenge
# https://www.kaggle.com/c/demand-forecasting-kernels-only
# !pip install lightgbm
# conda install lightgbm

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import lightgbm as lgb
from lightgbm import early_stopping
import warnings

pd.set_option('display.max_columns', None)
#pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


########################
# Loading the data
########################

train = pd.read_csv('Time_Series/datasets/demand_forecasting/train.csv', parse_dates=['date'])
test = pd.read_csv('Time_Series/datasets/demand_forecasting/test.csv', parse_dates=['date'])

sample_sub = pd.read_csv('Time_Series/datasets/demand_forecasting/sample_submission.csv')

df = pd.concat([train, test], sort=False)


#####################################################
# EDA
#####################################################

df["date"].min(), df["date"].max()

check_df(df)

df[["store"]].nunique()

df[["item"]].nunique()

df.groupby(["store"])["item"].nunique()

df.groupby(["store", "item"]).agg({"sales": ["sum"]})

df.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})

df.head()


#####################################################
# FEATURE ENGINEERING
#####################################################

df.head()

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.isocalendar().week
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)

df.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})


########################
# Random Noise
########################

def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))


########################
# Lag/Shifted Features
########################

df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)

pd.DataFrame({"sales": df["sales"].values[0:10],
              "lag1": df["sales"].shift(1).values[0:10],
              "lag2": df["sales"].shift(2).values[0:10],
              "lag3": df["sales"].shift(3).values[0:10],
              "lag4": df["sales"].shift(4).values[0:10]})

df.groupby(["store", "item"])['sales'].head()

df.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(1))

def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

# Tahmin edeceğimiz zaman gelecek 3 ay olduğu için gecikmeyi 3 ay gerisinden başlattık
df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])

df.head()
df.tail()

check_df(df)

########################
# Rolling Mean Features
########################

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].rolling(window=5).mean().values[0:10]})

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "roll3": df["sales"].shift(1).rolling(window=3).mean().values[0:10],
              "roll5": df["sales"].shift(1).rolling(window=5).mean().values[0:10]})


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

# 1 yıl ve 1.5 yıl önceki tüm değerlerin hareketli ortalaması
df = roll_mean_features(df, [365, 546])

########################
# Exponentially Weighted Mean Features
########################

pd.DataFrame({"sales": df["sales"].values[0:10],
              "roll2": df["sales"].shift(1).rolling(window=2).mean().values[0:10],
              "ewm099": df["sales"].shift(1).ewm(alpha=0.99).mean().values[0:10],
              "ewm095": df["sales"].shift(1).ewm(alpha=0.95).mean().values[0:10],
              "ewm07": df["sales"].shift(1).ewm(alpha=0.7).mean().values[0:10],
              "ewm02": df["sales"].shift(1).ewm(alpha=0.1).mean().values[0:10]})

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)
df.tail()
check_df(df)

########################
# One-Hot Encoding
########################

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])

check_df(df)

########################
# Converting sales to log(1+sales)
########################

df['sales'] = np.log1p(df["sales"].values)

check_df(df)

#####################################################
# Model
#####################################################

########################
# Custom Cost Function
########################

# MAE, MSE, RMSE, SSE

# MAE: mean absolute error
# MAPE: mean absolute percentage error
# SMAPE: Symmetric mean absolute percentage error (adjusted MAPE)

def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


########################
# Time-Based Validation Sets
########################

train
test

# 2017'nin başına kadar (2016'nın sonuna kadar) train seti.
train = df.loc[(df["date"] < "2017-01-01"), :]

# 2017'nin ilk 3'ayı validasyon seti.
val = df.loc[(df["date"] >= "2017-01-01") & (df["date"] < "2017-04-01"), :]

cols = [col for col in train.columns if col not in ['date', 'id', "sales", "year"]]

Y_train = train['sales']
X_train = train[cols]

Y_val = val['sales']
X_val = val[cols]

Y_train.shape, X_train.shape, Y_val.shape, X_val.shape

########################
# LightGBM ile Zaman Serisi Modeli
########################

# !pip install lightgbm
# conda install lightgbm


# LightGBM parameters
lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': -1,
              'num_boost_round': 1000,
              'early_stopping_rounds': 200,
              'nthread': -1}

callbacks = [lgb.early_stopping(stopping_rounds=lgb_params['early_stopping_rounds']),
             lgb.log_evaluation(period=100)]

# metric mae: l1, absolute loss, mean_absolute_error, regression_l1
# mse: l2, square loss, mean_squared_error, mse, regression_l2, regression
# rmse, root square loss, root_mean_squared_error, l2_root
# mape, MAPE loss, mean_absolute_percentage_error

# num_leaves: bir ağaçtaki maksimum yaprak sayısı
# learning_rate: shrinkage_rate, eta
# feature_fraction: rf'nin random subspace özelliği. her iterasyonda rastgele göz önünde bulundurulacak değişken sayısı.
# max_depth: maksimum derinlik
# num_boost_round: n_estimators, number of boosting iterations. En az 10000-15000 civarı yapmak lazım.

# early_stopping_rounds: validasyon setindeki metrik belirli bir early_stopping_rounds'da ilerlemiyorsa yani
# hata düşmüyorsa modellemeyi durdur.
# hem train süresini kısaltır hem de overfit'e engel olur.
# nthread: num_thread, nthread, nthreads, n_jobs

lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)

model = lgb.train(lgb_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=lgb_params['num_boost_round'],
                  feval=lgbm_smape,
                  callbacks=callbacks)

y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(Y_val)) # log alınan değerleri geri alma işlemi expm1


########################
# Değişken Önem Düzeyleri
########################

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show(block=True)
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=200)

plot_lgb_importances(model, num=30, plot=True)


feat_imp = plot_lgb_importances(model, num=200)

importance_zero = feat_imp[feat_imp["gain"] == 0]["feature"].values

imp_feats = [col for col in cols if col not in importance_zero]
len(imp_feats)


########################
# Final Model
########################

train = df.loc[~df.sales.isna()]
Y_train = train['sales']
X_train = train[cols]


test = df.loc[df.sales.isna()]
X_test = test[cols]

lgb_params = {'num_leaves': 10,
              'learning_rate': 0.02,
              'feature_fraction': 0.8,
              'max_depth': 5,
              'verbose': -1,
              'nthread': -1,
              "num_boost_round": model.best_iteration}

lgbtrain_all = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)

final_model = lgb.train(lgb_params, lgbtrain_all, num_boost_round=model.best_iteration)


test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

########################
# Submission File
########################

test.head()

submission_df = test.loc[:, ["id", "sales"]]
submission_df['sales'] = np.expm1(test_preds)

submission_df['id'] = submission_df.id.astype(int)

# submission_df.to_csv("submission_demand.csv", index=False)













































































































