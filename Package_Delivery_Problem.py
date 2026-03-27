#Εργασία 2 - Άσκηση 2 (AM: 2123101)
#Ονοματεπώνυμο: Βουγιουκλόγλου Ραφαήλ
#Εξάμηνο: 5ο

#Algorithm used: PSO

#libraries:
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

#########################################################################

#ΚΛΑΣΕΙΣ:

class paketo:
    def __init__(self, id, weight, width, length, value):
        self.id = id
        self.weight = weight
        self.width = width
        self.length = length
        self.value = value

        #θέση (x, y) του πακέτου 
        self.x = -1
        self.y = -1
        self.loaded = False

class fortigo:
    def __init__(self, id, max_weight, width, length):
        self.id = id
        self.max_weight = max_weight
        self.width = width
        self.length = length
        self.current_weight = 0     #τρέχον βάρος
        self.loaded_packages = []  #πίνακας με τα φορτωμένα πακέτα
    
    #καθαρισμός φορτηγού για νέα δοκιμή (απαραίτητο για PSO)
    def reset(self):
        self.current_weight = 0
        self.loaded_packages = []

#########################################################################

#ΜΗΧΑΝΙΣΜΟΣ ΤΟΠΟΘΕΤΗΣΗΣ

#έλεγχος για επικαλυψη:

def epikalipsi(new_package, placed_packages):

    #Ελέγχουμε αν το νέο πακετο "πέφτει" πάνω στο τοποθετημένο πακέτο (True αν γίνεται σύγκρουση)

    oxi_epikalipsi = (
        new_package.x + new_package.width <= placed_packages.x or      #p1 αριστερά του p2
        new_package.x >= placed_packages.x + placed_packages.width or  #p1 δεξιά του p2
        new_package.y + new_package.length <= placed_packages.y or     #p1 κάτω του p2
        new_package.y >= placed_packages.y + placed_packages.length    #p1 πάνω του p2
    )

    return not oxi_epikalipsi #true αν υπάρχει επικάλυψη

#τοποθέτηση πακέτου:

def fortosi(fortigo, paketo):
    
    #τοποθέτηση πακέτου με στρατηγική "bottom left" (candidate points)

    if fortigo.current_weight + paketo.weight > fortigo.max_weight:     #έλεγχος βάρους
        return False
    
    candidate_points = [(0, 0)]     #δημιουργία υποψήφιων σημείων...

    for p in fortigo.loaded_packages:
        candidate_points.append((p.x + p.width, p.y))  #...δεξιά και...
        candidate_points.append((p.x, p.y + p.length))  #..πάνω

        candidate_points.sort(key=lambda pt: (pt[1], pt[0]))    #ταξινόμηση (1. χαμηλά και 2. αριστερά)

    for (cx, cy) in candidate_points:
        paketo.x = cx
        paketo.y = cy

        if (cx + paketo.width > fortigo.width) or (cy + paketo.length > fortigo.length):
            continue    #έλεγχος κάθε σημείου αν βγαίνει εκτός ορίων φορτηγού

        collision = False
        for loaded_pk in fortigo.loaded_packages:       
            if epikalipsi(paketo, loaded_pk):       #έλεγχος σύγκρουσης με τα φορτωμένα πακέτα
                collision = True
                break   #βρέθηκε σύγκρουση, σταματάμε τον έλεγχο για αυτό το σημείο

        if not collision:
            fortigo.loaded_packages.append(paketo)
            fortigo.current_weight += paketo.weight     #αν δεν συγκρουστεί το πακέτο, μένει μέσα 
            paketo.loaded = True
            return True
        
    return False    #δεν βρέθηκε υπάρχον χώρος για το πακέτο
    
#########################################################################

#ΑΛΓΟΡΙΘΜΟΣ PSO

#σωματίδιο = μια λύση (οχι οι συντεταγμένες, αλλά η σειρά προτεραιότητας)

class Particle:
    def __init__(self, num_packages):
        self.position = [random.uniform(0, 1) for _ in range(num_packages)]     #τυχαίος αριθμός προτεραιότητας (τιμή από το 0 μεχρι 1)
        self.velocity = [random.uniform(-0.1, 0.1) for _ in range(num_packages)]       #ταχύτητα μεταβολής προτεραιοτήτων

        self.best_position = self.position[:]   #κρατάμε την καλύτερη θέση που βρέθηκε
        self.best_score = -1.0  #αρχικοποίηση

def fitness_function(particle_position, packages_list, trucks_list):

    #συνάρτηση αξιολόγησης - υπολογίζει πόσο καλή είναι μια λύση (particle) + επιστρέφει τη συνολική αξία

    for t in trucks_list:
        t.reset()       #καθαρισμός (reset)

    paired = zip(packages_list, particle_position)      #ζευγαρώνουμε τα πακέτα με τις τιμές προτεραιότητας σωματιδίου...
    sorted_packages = sorted(paired, key = lambda x: x[1], reverse = True)      #...και τα ταξινομούμε σε φθίνουσα σειρά (όσο μεγαλύτερος ο αριθμός, τοσο πιο νωρίς μπαίνει στο φορτηγό)

    total_value = 0

    for pk_obj, priority_val in sorted_packages:    #φόρτωση με σειρά προτεραιότητας
        loaded = False
        for truck in trucks_list:
            if fortosi(truck, pk_obj):
                loaded = True
                total_value += pk_obj.value #κέρδος (αξία) πακέτου προσμετράται
                break   #μπήκε στο φορτηγό, συνεχίζουμε στο επόμενο πακέτο
        
    return total_value
    
def run_pso(packages, trucks):

    #παράμετροι PSO
    num_particles = 30  #αριθμός λύσεων
    iterations = 100    #αριθμός βελτιώσεων
    w = 0.7   #αδράνεια 
    c1 = 1.5  #προσωπική γνώση
    c2 = 1.5  #γνώση σμήνους

    num_pk = len(packages)
    swarm = [Particle(num_pk) for _ in range(num_particles)]    #αρχικοποίηση σμήνους

    global_best_position = []
    global_best_score = -1.0

    print(f"\n[INFO] Εκκίνηση PSO με {num_particles} σωματίδια και {iterations} επαναλήψεις...")

    for i in range (iterations):
        for p in swarm:
            score = fitness_function(p.position, packages, trucks)   #υπολογισμός score (fitness)

            if score > p.best_score:     #ενημέρωση personal best
                p.best_score = score     
                p.best_position = p.position[:]

            if score > global_best_score:       #ενημέρωση global best
                global_best_score = score
                global_best_position = p.position[:]

        for p in swarm:      #ενημέρωση θέσης και ταχύτητας σωματιδίων
            for d in range(num_pk):
                r1 = random.random()
                r2 = random.random()

                #τύπος ταχύτητας PSO
                vel_cognitive = c1 * r1 * (p.best_position[d] - p.position[d])
                vel_social = c2 * r2 * (global_best_position[d] - p.position[d])
                p.velocity[d] = (w * p.velocity[d] + vel_cognitive + vel_social)

                p.position[d] += p.velocity[d]    #ενημέρωση θέσης

    return global_best_position, global_best_score

#########################################################################

#ΣΥΝΕΧΕΙΑ PSO ΜΕ ΕΙΣΟΔΟΥΙ ΤΟΥ ΧΡΗΣΤΗ

def get_input_int(message):
    while True:
        try:
            val = int(input(message))
            if val < 0: print("Πρέπει να είναι θετικός αριθμός!"); continue
            return val
        except ValueError:
            print("Μη έγκυρη είσοδος! Πρέπει να ειναι αριθμός.")

def get_input_float(message):
    while True:
        try:
            val = float(input(message))
            if val < 0: print("Πρέπει να ειναι θετικός αριθμός!"); continue
            return val
        except ValueError:
            print("Μη έγκυρη είσοδος! Πρέπει να ειναι αριθμός.")

def main():
    print("Εκκίνηση αλγορίθμου PSO")

    num_trucks = get_input_int("Πλήθος φορτηγών: ")     #εισαγωγή φορτηγών
    trucks = []
    for i in range(num_trucks):
        print(f"\nΦορτηγό {i+1}")
        mw = get_input_float("Μέγιστο βάρος: ")
        w = get_input_float("Πλάτος καμπίνας: ")
        l = get_input_float("Μήκος καμπίνας: ")
        trucks.append(fortigo(i+1, mw, w, l))

    num_packages = get_input_int("\nΠλήθος πακέτων: ")      #εισαγωγή πακέτων
    packages = []
    for i in range(num_packages):
        print(f"\nΠακέτο {i+1}")
        w_pk = get_input_float("Βάρος: ")
        width = get_input_float("Πλάτος: ")
        length = get_input_float("Μήκος: ")
        val = get_input_float("Τιμή (κόστος μεταφοράς): ")
        packages.append(paketo(i+1, w_pk, width, length, val))

    best_solution_pos, best_val = run_pso(packages, trucks)     #εκτέλεση αλγορίθμου PSO (επιστρέφει τις προτεραιότητες!!!)

    print("\n")
    print(f"Βέλτιστη λύση (συνολική αξία {best_val})")      #εμφάνιση αποτελέσματος
    print("\n")

    fitness_function(best_solution_pos, packages, trucks)   #τρέχουμε ξανά το fitness function για να γεμίσουν τα φορτηγά και να τυπώσουμε το τελικό αποτέλεσμα

    grand_total_packages = 0

    for t in trucks:
        print(f"[Φορτηγό {t.id}] (Φορτίο: {t.current_weight}/{t.max_weight})")
        if not t.loaded_packages:
            print("Κενό")
        else:
            for p in t.loaded_packages:
                print(f"Πακέτο {p.id}: Θέση(x = {p.x}, y = {p.y}, Διαστ({p.width} x {p.length}), Αξία = {p.value}")
                grand_total_packages += 1
    
    print(f"\nΣυνοπτικά: ")
    print(f"Συνολικά πακέτα προς αποστολή: {len(packages)}")
    print(f"Πακέτα που χωρέσαν: {grand_total_packages}")
    print(f"Πακέτα που έμειναν εκτός: {len(packages) - grand_total_packages}")  

if __name__ == "__main__":
    main()    
