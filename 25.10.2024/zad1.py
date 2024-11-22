#pregnant-times,glucose-concentr,blood-pressure,
#skin-thickness,insulin,mass-index,pedigree-func,age,class

#6,148,72,35,0,33.6,0.627,50,tested_positive

import numpy as np

def forwardPass(wiek, waga, wzrost):


    w1 = [-0.46122, 0.78548]
    w2 = [ 0.97314, 2.10584]
    w3 = [-0.39203, -0.57847]
    bias_h1 = 0.80109
    bias_h2 = 0.43529


    hidden1 = wiek * w1[0] + waga * w2[0] + wzrost * w3[0] + bias_h1
    hidden1_po_aktywacji = 1 / (1 + np.exp(-hidden1)) 
    hidden2 = wiek * w1[1] + waga * w2[1] + wzrost * w3[1] + bias_h2
    hidden2_po_aktywacji = 1 / (1 + np.exp(-hidden2))

    output_weights = [-0.81546, 1.03775]
    output_bias = -0.2368
    output = hidden1_po_aktywacji * output_weights[0] + hidden2_po_aktywacji * output_weights[1] + output_bias
    
    return output

# Test the function
print(forwardPass(23, 75, 176))
print(forwardPass(25, 67, 180))
