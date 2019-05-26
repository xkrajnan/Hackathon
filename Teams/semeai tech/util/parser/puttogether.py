one_long_string = ''

for i in range(2, 40):
    title = 'parsed_amodelrecommends' + str(i) + '.txt'
    
    with open(title, 'r') as infile:
        one_long_string = one_long_string + infile.read()

with open('whole_parsed.txt', 'w') as outfile:
    outfile.write(one_long_string)
