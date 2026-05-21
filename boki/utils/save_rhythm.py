import numpy as np

def save_rhythm(start_point, end_point, label, filepath):
    rhythm = []
    total_time_0 = 0
    total_time_1 = 0
    total_time_2 = 0
    total_time_3 = 0

    for i in range(len(start_point)):
      # print(rhythm)
      if(i==0 or label[0][i] != rhythm[-1][1] or i == len(start_point)):
        rhythm.append([start_point[i], label[0][i]])
        if(len(rhythm) > 1 and rhythm[-2][1] == 0):
          total_time_0 += (rhythm[-1][0] - rhythm[-2][0])
        if(len(rhythm) > 1 and rhythm[-2][1] == 1):
          total_time_1 += (rhythm[-1][0] - rhythm[-2][0])
        if(len(rhythm) > 1 and rhythm[-2][1] == 2):
          total_time_2 += (rhythm[-1][0] - rhythm[-2][0])
        if(len(rhythm) > 1 and rhythm[-2][1] == 3):
          total_time_3 += (rhythm[-1][0] - rhythm[-2][0])
      
    with open(filepath+"/rhythm.csv", 'w', newline='\n') as f:
        for i in range(len(rhythm)):
            if(i!=len(rhythm)-1): 
              f.write(str(int(rhythm[i][0])) + ',' + str(int(rhythm[i+1][0]-1)) + ',' + str(int(rhythm[i][1])) + '\n')
            else: 
               f.write(str(int(rhythm[i][0])) + ',' + str(int(end_point[-1]-1)) + ',' + str(int(rhythm[i][1])) + '\n')
               print('number_of_windows: ' + str(len(start_point)))

    acc = (total_time_0/360)/(len(start_point)*5)

    print("\n===================================")
    print("N: " + str(total_time_0))
    print("AF: " + str(total_time_1))
    print("VF: " + str(total_time_2))
    print("SVTA: " + str(total_time_3))
    print('accuracy: ', acc)
    print("===================================\n")
    