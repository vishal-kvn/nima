import csv

ava_filename = 'AVA.txt'
csv_filename = 'labels.csv'

with open(csv_filename, 'w') as csvfile:  
  csvwriter = csv.writer(csvfile)
  csvwriter.writerow(['image_name', 'tags'])

  with open(ava_filename) as fp:
    lines = fp.readlines()
    for line in lines:
      line = line.split(' ')
      csvwriter.writerow([line[1], ' '.join(line[2:12])])
