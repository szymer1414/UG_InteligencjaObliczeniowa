#Korekta stworzona na podstawie mojego rozwiazania
from datetime import datetime
import math

def calculate_age(birthdate):
    today = datetime.today()
    age_in_days = (today - birthdate).days

    # Calculate age in years
    years = today.year - birthdate.year
    if (today.month, today.day) < (birthdate.month, birthdate.day):
        years -= 1

    # Calculate days since last birthday
    last_birthday = datetime(today.year if today >= datetime(today.year, birthdate.month, birthdate.day) else today.year - 1, birthdate.month, birthdate.day)
    days_since_birthday = (today - last_birthday).days

    print(f"Masz {years} lat i {days_since_birthday} dni. Dzisiaj twój {age_in_days} dzień życia!")

    return age_in_days

# Input for name and birthdate
name = input("Podaj imię: ")
birthdate_input = input("Data urodzenia (YYYY.MM.DD): ")
birthdate = datetime.strptime(birthdate_input, "%Y.%m.%d")

print(f"Cześć {name}!")
days_old = calculate_age(birthdate)

# Biorhythm calculations
fizyczna = math.sin((2 * math.pi) / 23 * days_old)
emocjonalna = math.sin((2 * math.pi) / 28 * days_old)
intelektualna = math.sin((2 * math.pi) / 33 * days_old)

# Print biorhythm values
print("Fizyczna:", fizyczna)
print("Emocjonalna:", emocjonalna)
print("Intelektualna:", intelektualna)

# Calculate the average of biorhythms
srednia = (fizyczna + emocjonalna + intelektualna) / 3

# Check the average and display encouraging message
if srednia > 0.5:
    print("No i elegancko, tak trzymaj byczq!")
else:
    # If the average is low, calculate for the next day and compare
    fizyczna_next = math.sin((2 * math.pi) / 23 * (days_old + 1))
    emocjonalna_next = math.sin((2 * math.pi) / 28 * (days_old + 1))
    intelektualna_next = math.sin((2 * math.pi) / 33 * (days_old + 1))
    
    next_day_avg = (fizyczna_next + emocjonalna_next + intelektualna_next) / 3
    
    if next_day_avg > srednia:
        print("Nie martw się bracie, jutro będzie lżej,")
        print("a nawet jak nie będzie, to i tak się dzisiaj śmiej!")
    print(f"Dzisiejsza średnia: {srednia:.2f}")
    print(f"Jutrzejsza średnia: {next_day_avg:.2f}")

# Informational comment
# Napisanie tego programu zajęło mi 1h
