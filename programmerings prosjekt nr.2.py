

def main():
    
    correct_awnsers = ("'b. Oslo'", "'c. Krone'", "'a. Oslo'", "'b. 17th May'", "'a. Red'", "'c. 3'", "'d. NTNU'", "'b. 196 km'", "'c. South-west'", "'b.  Bergen'")
    
    questions = ("'What is the capital of Norway?'", "'What is the currency of Norway'", "'What is the largest city in Norway?'", 
                 "'When is constitution day (the national day) of Norway?'", "'What color is the background of the Norwegian flag?'",
                 "'How many countries does Norway border?'", "'What is the name of the university in Trondheim?'",
                 "'How long is the border between Norway and Russia?'", "'Where in Norway is Stavanger?'",
                 "'From which Norwegian city did the world famous composer Edvard Grieg come?'")

    
    result = [quiz_q1(), quiz_q2(), quiz_q3(), quiz_q4(), quiz_q5(), quiz_q6(), quiz_q7(), quiz_q8(), quiz_q9(), quiz_q10()]
    
    awnser = [q1_awnser, q2_awnser, q3_awnser, q4_awnser, q5_awnser, q6_awnser, q7_awnser, q8_awnser, q9_awnser, q10_awnser]
    
    correct_prosent = 0
    
    length = 0
    
    for item in result:

        
        if item == True:
            correct_prosent = correct_prosent + 10
            
            
        else:
            print("")
            print(f"you awnserd {questions[length]} wrong")
            print("")
            print(f"you awnser was '{awnser[length]}'")
            print("")
            print(f"the correct awnser was {correct_awnsers[length]}")
            print("")
        
        length = length + 1     
    print(f"you had {correct_prosent}% correct")   




def login_info():
    

    
    username_and_password = {
        "user_name" :"MEK1300",
        
        "password" : "Python"
  
        }
        
    
    while True:
        
        username_entery =  input("enter user name: ")
        
        
        password_emtery = input("enter password: ")
        
        
        if username_entery == username_and_password["user_name"] and password_emtery == username_and_password["password"]:
           
            print("you enterd correct password")
            print("")
            
            break
        
        else: 
            
            print("")
            print("Invalid username and/or password")
            print("")
        
login_info()     


def quiz_q1():
    print("")
    print("What is the capital of Norway")
    print("")
    print("a. Bergen")
    print("b. Oslo")
    print("c. Stavanger")
    print("d. Trondheim")
    
    
    while True:
        user_guess = input("enter your awnser(a, b, c or d): ")
    
        global q1_awnser 
    
        
            
        if user_guess.upper() == "A":
            q1_result = False 
            q1_awnser = "a. Bergen"
            break
                
        elif user_guess.upper() == "B":
            
            q1_result = True
            q1_awnser = "b. Oslo"
            break
                
        elif user_guess.upper() == "C":
            
            q1_result = False
            q1_awnser = "c. Stavanger"
            break
                
        elif user_guess.upper() == "D":
            
            q1_result = False
            q1_awnser = "d. Trondheim"
            break
               
           
                
        else:
            print("not a valid awnser")
 
    
    
    
    return q1_result 
            
        
def quiz_q2():
    print("")
    print("What is the currency of Norway?")
    print("")
    print("a. Euro")
    print("b. Pound")
    print("c. Krone")
    print("d. Deutsche Mark")
    
    
    while True:
        user_guess = input("enter your awnser(a, b, c or d): ")
    
        
        global q2_awnser 
        
            
        if user_guess.upper() == "A":
            q2_result = False 
            q2_awnser = "a. Euro"
            break
                
        elif user_guess.upper() == "B":
            
            q2_result = False
            q2_awnser = "b. Pound"
            break
                
        elif user_guess.upper() == "C":
            
            q2_result = True
            q2_awnser = "c. Krone"
            break
                
        elif user_guess.upper() == "D":
            
            q2_result = False
            q2_awnser = "d. Deutsche Mark"
            break
               
           
                
        else:
            print("not a valid awnser")
 
    
    
    
    return q2_result 
                    

def quiz_q3():
    print("")
    print("What is the largest city in Norway?")
    print("")
    print("a. Oslo")
    print("b. Stavanger")
    print("c. Bergen")
    print("d. Trondheim")
    
    
    while True:
        user_guess = input("enter your awnser(a, b, c or d): ")
    
        
        global q3_awnser 
        
            
        if user_guess.upper() == "A":
            q3_result = True
            q3_awnser = "a. Oslo"
            break
                
        elif user_guess.upper() == "B":
            
            q3_result = False
            q3_awnser = "b. Stavanger"
            break
                
        elif user_guess.upper() == "C":
            
            q3_result = False 
            q3_awnser = "c. Bergen"
            break
                
        elif user_guess.upper() == "D":
            
            q3_result = False
            q3_awnser = "d. Trondheim"
            break
               
           
                
        else:
            print("not a valid awnser")
 
    
    
    
    return q3_result   


def quiz_q4():
    print("")
    print("When is constitution day (the national day) of Norway?")
    print("")
    print("a. 27th May")
    print("b. 17th May")
    print("c. 17th April")
    print("d. 27th April")
    
    
    while True:
        user_guess = input("enter your awnser(a, b, c or d): ")
    
        
        global q4_awnser 
        
            
        if user_guess.upper() == "A":
            q4_result = False 
            q4_awnser = "a. 27th May"
            break
                
        elif user_guess.upper() == "B":
            
            q4_result = True 
            q4_awnser = "b. 17th May"
            break
                
        elif user_guess.upper() == "C":
            
            q4_result = False 
            q4_awnser = "c. 17th April"
            break
                
        elif user_guess.upper() == "D":
            
            q4_result = False
            q4_awnser = "d. 27th April"
            break
               
           
                
        else:
            print("not a valid awnser")
 
    
    
    
    return q4_result   


def quiz_q5():
    print("")
    print("What color is the background of the Norwegian flag?")
    print("")
    print("a. Red")
    print("b. White")
    print("c. Blue")
    print("d. Yellow")
    
    
    while True:
        user_guess = input("enter your awnser(a, b, c or d): ")
    
        
        global q5_awnser 
    
        
            
        if user_guess.upper() == "A":
            q5_result = True
            q5_awnser = "a. Red"
            break
                
        elif user_guess.upper() == "B":
            
            q5_result = False
            q5_awnser = "b. White"
            break
                
        elif user_guess.upper() == "C":
            
            q5_result = False 
            q5_awnser = "c. Blue"
            break
                
        elif user_guess.upper() == "D":
            
            q5_result = False
            q5_awnser = "d. Yellow"
            break
               
           
                
        else:
            print("not a valid awnser")
 
    
    
    
    return q5_result 


def quiz_q6():
    print("")
    print("How many countries does Norway border?")
    print("")
    print("a. 1")
    print("b. 2")
    print("c. 3")
    print("d. 4")
    
    
    while True:
        user_guess = input("enter your awnser(a, b, c or d): ")
    
        
        global q6_awnser 
        
            
        if user_guess.upper() == "A":
            q6_result = False
            q6_awnser = "a. 1"
            break
                
        elif user_guess.upper() == "B":
            
            q6_result = False
            q6_awnser = "b. 2"
            break
                
        elif user_guess.upper() == "C":
            
            q6_result = True
            q6_awnser = "c. 3"
            break
                
        elif user_guess.upper() == "D":
            
            q6_result = False
            q6_awnser = "d. 4"
            break
               
           
                
        else:
            print("not a valid awnser")
 
    
    
    
    return q6_result 


def quiz_q7():
    print("")
    print("What is the name of the university in Trondheim?")
    print("")
    print("a. UiS")
    print("b. UiO")
    print("c. NMBU")
    print("d. NTNU")
    
    
    while True:
        user_guess = input("enter your awnser(a, b, c or d): ")
    
        
        global q7_awnser 
        
            
        if user_guess.upper() == "A":
            q7_result = False
            q7_awnser = "a. UiS"
            break
                
        elif user_guess.upper() == "B":
            
            q7_result = False
            q7_awnser = "b. UiO"
            break
                
        elif user_guess.upper() == "C":
            
            q7_result = False
            q7_awnser = "c. NMBU"
            break
                
        elif user_guess.upper() == "D":
            
            q7_result = True
            q7_awnser = "d. NTNU"
            break
               
           
                
        else:
            print("not a valid awnser")
 
    
    
    
    return q7_result


def quiz_q8():
    print("")
    print("How long is the border between Norway and Russia?")
    print("")
    print("a. 96 km")
    print("b. 196 km")
    print("c. 296 km")
    print("d. 396 km")
    
    
    while True:
        user_guess = input("enter your awnser(a, b, c or d): ")
    
        
        global q8_awnser  
        
            
        if user_guess.upper() == "A":
            q8_result = False
            q8_awnser = "a. 96 km"
            break
                
        elif user_guess.upper() == "B":
            
            q8_result = True 
            q8_awnser = "b. 196 km"
            break
                
        elif user_guess.upper() == "C":
            
            q8_result = False
            q8_awnser = "c. 296 km"
            break
                
        elif user_guess.upper() == "D":
            
            q8_result = False 
            q8_awnser = "d. 396 km"
            break
               
           
                
        else:
            print("not a valid awnser")
 
    
    
    
    return q8_result


def quiz_q9():
    print("")
    print("Where in Norway is Stavanger?")
    print("")
    print("a. North")
    print("b. South")
    print("c. South-west")
    print("d. South-east")
    
    
    while True:
        user_guess = input("enter your awnser(a, b, c or d): ")
    
        
        global q9_awnser 
        
            
        if user_guess.upper() == "A":
            q9_result = False
            q9_awnser = "a. North"
            break
                
        elif user_guess.upper() == "B":
            
            q9_result = False 
            q9_awnser = "b. South"
            break
                
        elif user_guess.upper() == "C":
            
            q9_result = True 
            q9_awnser = "c. South-west"
            break
                
        elif user_guess.upper() == "D":
            
            q9_result = False 
            q9_awnser = "d. South-east"
            break
               
           
                
        else:
            print("not a valid awnser")
 
    
    
    
    return q9_result


def quiz_q10():
    print("")
    print("From which Norwegian city did the world famous composer Edvard Grieg come?")
    print("")
    print("a. Oslo")
    print("b. Bergen")
    print("c. Stavanger")
    print("d. Tromsø")
    
    
    while True:
        user_guess = input("enter your awnser(a, b, c or d): ")
    
        
        global q10_awnser 
        
            
        if user_guess.upper() == "A":
            q10_result = False
            q10_awnser = "a. Oslo"
            break
                
        elif user_guess.upper() == "B":
            
            q10_result = True  
            q10_awnser = "b. Bergen"
            break
                
        elif user_guess.upper() == "C":
            
            q10_result = False 
            q10_awnser = "c. Stavanger"
            break
                
        elif user_guess.upper() == "D":
            
            q10_result = False 
            q10_awnser = "d. Tromsø"
            break
               
           
                
        else:
            print("not a valid awnser")
 
    
    
    
    return q10_result

main()