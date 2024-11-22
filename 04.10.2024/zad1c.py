#prgoram w calosci wygenerowany przez OpenAI na bazie moich instrukcji
import math
from datetime import datetime, timedelta

# Function to calculate number of days alive
def calculate_days_alive(birthdate):
    today = datetime.now()
    birthdate = datetime.strptime(birthdate, "%Y-%m-%d")
    days_alive = (today - birthdate).days
    return days_alive

# Function to calculate the bio-rhythm values
def calculate_biorhythms(days_old):
    fiz = math.sin((2 * math.pi / 23) * days_old)
    emo = math.sin((2 * math.pi / 28) * days_old)
    intell = math.sin((2 * math.pi / 33) * days_old)
    return fiz, emo, intell

# Main function
def main():
    # Get user input
    name = input("Enter your name: ")
    birthdate = input("Enter your birthdate (YYYY-MM-DD): ")
    
    # Calculate days alive
    days_alive = calculate_days_alive(birthdate)
    
    # Calculate today's biorhythms
    fiz, emo, intell = calculate_biorhythms(days_alive)
    average_today = (fiz + emo + intell) / 3
    
    # Display results
    print(f"\n{name}, you've been alive for {days_alive} days.")
    print(f"Today's biorhythms:")
    print(f"Physical (fiz): {fiz:.2f}")
    print(f"Emotional (emo): {emo:.2f}")
    print(f"Intellectual (int): {intell:.2f}")
    print(f"Average for today: {average_today:.2f}\n")
    
    # Check if today is a good day or not
    if average_today > 0.5:
        print("It must be a good day!")
    elif average_today < -0.5:
        # Calculate tomorrow's biorhythm average
        fiz_tomorrow, emo_tomorrow, intell_tomorrow = calculate_biorhythms(days_alive + 1)
        average_tomorrow = (fiz_tomorrow + emo_tomorrow + intell_tomorrow) / 3
        if average_tomorrow > average_today:
            print("Today may be tough, but tomorrow the sun will shine brighter!")
        else:
            print("Tomorrow may not be much better, hang in there!")

if __name__ == "__main__":
    main()
