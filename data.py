with open('train.csv', 'r') as file1:
    lines = file1.readlines()
    with open('train_bin.csv', 'w') as file2:
        for line in lines:
            lst = line.split(",")
            output = []
            for i in lst[1:]:
                if int(i) < 127.5:
                    output.append(0)
                else:
                    output.append(1)
            file2.write(f"{lst[0]}, {str(output).replace('[', '').replace(']', '')}\n")