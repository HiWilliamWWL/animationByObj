import glob
#files = glob.glob("./rawData/newDataTotal/2020-9-24/*.csv")
files = glob.glob("./dataPlace/raw/*.csv")
for fileName in files:
    nameF2 = "./dataPlace/process/" + fileName.split("/")[-1]
    print(nameF2)
    nameF1 = fileName
    with open(nameF1, "r+") as f1:
        line1 = f1.readline()
        line1 = f1.readline()
        line1 = f1.readline().split(",")
        line2 = f1.readline().split(",")
        line3 = f1.readline().split(",")
        line4 = f1.readline().split(",")
        line5 = f1.readline().split(",")
        resultField = ""
        for i, content in enumerate(line1):
            resultField += content.strip() + " " + line2[i].strip()   + " " + line3[i].strip()   + " " + line4[i].strip()   + " " + line5[i].strip() + ","
        resultField = resultField[:-1]
        with open(nameF2, "w+") as f2:
            f2.writelines(resultField)
            line = f1.readline()
            while line:
                f2.writelines(line)
                line = f1.readline()
            f2.close()
        f1.close()