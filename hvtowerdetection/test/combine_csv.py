import os
import csv

# path = 'E:/DeepMaze/bitbucket_repos/palm-tree-detection/testing-pipeline/cocoa-tree/results/csv_files'
path = 'results/csv_files'

# get list of immediate files in a directory
def get_subfiles(dir):
    "Get a list of immediate subfiles"
    return next(os.walk(dir))[2]

files = get_subfiles(path)
with open('towers_combined.csv', 'w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerow(['index', 'X_img', 'Y_img', 'latitude', 'longitude'])
    c = 1
    for filename in files:
        filepath = os.path.join(path, filename)
        print('\nfile: {}'.format(filename))
        dx = int(filename.split('.')[0].split('_')[1])
        dy = int(filename.split('.')[0].split('_')[2])
        with open(filepath, 'r') as fd:
            csv_reader = csv.reader(fd, delimiter=',')
            for row in csv_reader:
                if row[0].lower().strip() == 'index':
                    continue
                print('row: {}, {}, {}, {}, {}'.format(row[0], str(int(row[1])+dx), str(int(row[2])+dy), row[3], row[4]))
                csv_writer.writerow([row[0], str(int(row[1])+dx), str(int(row[2])+dy), str(row[3]), str(row[4])])
            fd.close()
        c += 1
    f.close()

    print('\nN files: {}'.format(c))
