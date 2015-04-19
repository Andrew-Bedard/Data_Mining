
lines = [x.split(";",1) for x in open("SmsCollection.csv")]

f = open("clean_sms_collection.csv", "w")
f.write("label;text\n")

for line in lines[1:]:
    text = line[1][:-1].replace('"', '\\"').replace("\\","\\")
    f.write(line[0]+";"+'"'+text+'"\n')
f.close()
