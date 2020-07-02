# importing os module 
import os 
  
# Function to rename multiple files 
def main(): 
  
    # print(len(os.listdir("bar clamp")))
    class_names = ['bar clamp', 'gear box', 'nozzle', 'part1', 'part3', 'pawn', 'turbine housing', 'vase']
    class_name = class_names[1]

    directory = "D:/Tensorflow2/Teste de treinamento/dataset/"
    for count, filename in enumerate(os.listdir(directory + class_name)): 
        # print(filename)
        new_count = format(count + 160, "03d")  # numera de 000 até a quantidade de arquivos
        new_filename = str(new_count) + ".jpg" # nomeia novo arquivo 000.jpg por exemplo
        dst = directory + class_name + "/" + new_filename 
        # print(dst)

        src = directory + class_name + "/" + filename # source address
        print(src)

        os.rename(src, dst) 
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 