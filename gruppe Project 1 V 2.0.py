#improtere en random modul fra ramdom bibloteket.
from random import randint 

print("")
print("Welcome to the gussing game, you have 5 attempts to guess a number bweteen 1 and 1000, good luck")
print("************************************************************************************************")
attempt = 1

while True:

    random_number = randint(1, 1000) 
        
    
    while attempt <= 5:
        user_guess = (input("Enter what you think the number is: "))
    
        if user_guess.isdigit() == True:
            user_guess = int(user_guess)
            
            if user_guess == random_number:
                print(f"Congratulation! You guessed the number: {random_number} in {attempt} attempt(s) \n")
                 
                attempt = 1
                break  
            
            elif attempt >= 5:
                print(f"You have run out of attempts, better luck next time \n the number was: {random_number}")
            
                attempt = 1
                break
            

            
            elif user_guess != random_number:
                
                if user_guess > random_number:
                    print("The number is too high \n") 
                 
                else:
                    print("The number is too low \n") 
                

                attempt_left = 5 - attempt 

                attempt = attempt + 1
           
                print(f"you have {attempt_left} try(s) left. \n") 
                
                continue
               

        else:
            print("not a vail number. please try again \n ")
                
            print("you did not use a attempt \n")      
    
                     
             
    counter = 0 
    while counter != 10: 
        user_input = input("Do you want to play again (enter yes if you want to play, no if not): ") 
        

        if user_input.upper() == "YES":
            break 
             
        elif user_input.upper() == "NO":
            print("")
            print("Hope to see you back soon.")
            counter = 10 
        else: 
            print("not a valid awnser")
            
    if counter == 10:
        break
       
    
       
   
    
     
        
   
   

     

  