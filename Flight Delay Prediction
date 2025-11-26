#Άσκηση 2 (AM: 2123101)
#Ονοματεπώνυμο: Βουγιουκλόγλου Ραφαήλ
#Εξάμηνο: 5ο

#Σύνολο δεδομένων που χρησιμοποιήθηκε: https://www.kaggle.com/datasets/izumita/flight-delay-sample

#libraries:
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

##########################################################################

#1ST PART OF ASSIGNMENT: ROUTE DELAY PREDICTION FROM USER INPUT

dataset = pd.read_csv('flight_data.csv')   #φόρτωση dataset

dataset = dataset.dropna(subset=[dataset.columns[15]])  #ignoring the empty values in the ArrDelay column

x = dataset.iloc[:, [9, 17, 18]].values     #airline, origin, destination
y = dataset.iloc[:, 15].values      #delay

x = np.column_stack((x, dataset['Month'], dataset['DayofMonth']))   #we add Month and DayofMonth to the training model

hasadded = 0    #for DepTime stack problem (line 108)

le1 = LabelEncoder()    #transform airline
x[:,0] = le1.fit_transform(x[:,0])
le2 = LabelEncoder()    #transform origin
x[:,1] = le2.fit_transform(x[:,1])
le3 = LabelEncoder()    #transform destination
x[:,2] = le3.fit_transform(x[:,2])

model = LinearRegression()  #training the model
model.fit(x, y)

while True: 
    ori = input("Enter origin airport (e.g. JFK, LAX, PHX etc.) ((or type 'exit' to exit)): ")    #type origin airport code

    if ori == "exit":   #the user can keep entering different routes until he types 'exit'
        print("\nExiting...")
        break

    if ori not in le2.classes_:
        print(f"Unknown origin airport! Try again.")
        continue
    
    while True:     
        des = input("Enter destination airport (e.g. MIA, SEA ,HOU etc.): ")   #type destination airport code

        if des not in le3.classes_:
            print(f"Unknown destination airport! Try again. ")
            continue
        else:
            break

    while True:
        air = input("Enter airline (e.g. AA, HP, US etc.): ")  #type airline ΙΑΤΑ code

        if air not in le1.classes_:
            print(f"Unknown airline! Try again.")
            continue
        else:
            break
    
    while True:
        try:
            month = int(input("Enter month (1-12): "))   #type month

            if month < 1 or month > 12:
                print("Wrong input for month! Try again.")
                continue
            else:
                break
        except ValueError:
            print("Wrong input! Month must be an integer. Try again.")
            continue

    while True:
        try:
            day = int(input("Enter day (1-31): "))     #type day

            if day < 1 or day > 31:
                print("Wrong input for day! Try again.")
                continue
            else:
                break
        except ValueError:
            print("Wrong input! Day must be an interger. Try again.")
            continue

    air_enc = le1.transform([air])      #transforming the airline value
    ori_enc = le2.transform([ori])      #transforming the origin value
    des_enc = le3.transform([des])      #transforming the destination value

    x_new = np.array([[air_enc[0], ori_enc[0], des_enc[0], month, day]])  #new value of x
    y_new = model.predict(x_new)    #prediction of the new y using the new x value

    print(f"\nPredicted delay for {ori} -> {des} with airline {air} on {day}/{month}: {y_new[0]:.2f} minutes\n")    #print the expected delay

    ##########################################################################

    #2ND PART OF ASSIGNMENT: LONG DELAY PREDICTION

    if hasadded == 0:     #so that we dont stack it again if the user searches for other routes
        x = np.column_stack((x, dataset['DepTime']))   #we add DepTime to the training model  
        hasadded = 1; 

    poly = PolynomialFeatures(degree=2)     #using polynomial regression for more detailed prediction
    x_poly = poly.fit_transform(x)

    model1 = LinearRegression()     #training a new linear regression model with a new x value
    model1.fit(x_poly, y)

    while True:  

        answer = input(f"Do you want to see days and hours with expected long delays on month {month} for this route? (yes/no): ")

        if answer == "yes":

            threshold = dataset['ArrDelay'].quantile(0.82)      #defining threshold as the top 18% most delayed flights in dataset

            found_delay = False

            for day1 in range (1, 31):          #searching the hours and days one by one
                for hour in range (0, 2400, 100):

                    X_new = np.array([[air_enc[0], ori_enc[0], des_enc[0], month, day1, hour]])

                    X_new_poly = poly.transform(X_new)

                    long_delay = model1.predict(X_new_poly)[0]

                    if long_delay > threshold:         #if long delay is above the threshold, it is printed on the screen
                        hours_new = hour // 100
                        minutes_new = hour % 100
                        print(f"\n• Day {day1}, Hour {int(hours_new):02d}:{int(minutes_new):02d}-> Predicted long delay: {long_delay:.2f} minutes\n")
                        found_delay = True

            if not found_delay:     #if long delay is below the threshold, this message appears
                print("\nNo long delays predicted for this route\n")        
           
            print("Going back to route selection...\n")     #we "break" from the inside while loop and we continue from the outside while loop
            break
        else:
            if answer == "no":
                print("\nExiting long delay search...\n")   #if user answers no, the program continues by asking a new route
                break
            else:
                print("Wrong input! Try again.")
                continue
