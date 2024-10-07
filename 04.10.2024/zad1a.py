#zadanie zrobine samodzielnie
from datetime import datetime
import math

def calculate_age(birthdate):
    today = datetime.today()
    age_in_days = (today - birthdate).days
    year = today.year - birthdate.year

    if (today.month, today.day) < (birthdate.month, birthdate.day):
        year -= 1
    last_birthday = datetime(today.year if today >= datetime(today.year, birthdate.month, birthdate.day) else today.year - 1, birthdate.month, birthdate.day)
    days = (today - last_birthday).days
    
    print(f"Masz {year} lat i {days} dni. Dzisiaj twój {age_in_days} dzień!")

    return age_in_days
    
name = input("Podaj imię:")



birthdate_input = input("Data urodzenia (YYYY.MM.DD): ")
birthdate = datetime.strptime(birthdate_input, "%Y.%m.%d")
print("Cześć " + name)
daysold=calculate_age(birthdate)

fiz= math.sin((2*math.pi)/(23)*daysold)
emo= math.sin((2*math.pi)/(28)*daysold)
int=math.sin((2*math.pi)/(33)*daysold)

print("fizyczna")
print(fiz)
print("Emocjonlana")
print(emo)
print("Intelektualna")    
print(int)
      
srednia = (fiz+int+emo)/3
if(srednia>0.5):
    print("no i elegancko, tak trzymac byczq")
else:
    fiz= math.sin((2*math.pi)/(23)*(daysold+1))
    emo= math.sin((2*math.pi)/(28)*(daysold+1))
    int=math.sin((2*math.pi)/(33)*(daysold+1))
    if(srednia<(fiz+int+emo)/3):
        print(" ")
        print(srednia)
        print((fiz+int+emo)/3)
        print("Nie martw się bracie, jutro będzie lżej,") 
        print("a nawet jak nie będzie, to i tak się dzisiaj śmiej")
    
        
        
#napisanie tego programu zajelo mi 1h
        
    
