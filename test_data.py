with open("myfile.csv","w") as file:
    for i in range(10):
        file.write(f"{i}\n")