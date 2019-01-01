f = open("data.tsv", "r")
lines = f.readlines()
counter = 0
f1 = open("sample.tsv", "w")
f2 = open("validation.tsv", "w")

for line in lines:
	if counter <= 300000:
		f1.write(line)
	elif counter > 300000 and counter < 600000:
		f2.write(line)
	counter+=1

f.close()
f1.close()
f2.close()
