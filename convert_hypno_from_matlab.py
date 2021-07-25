import scipy.io as sio


def get_id(id):
    # wake
    if id == 11:
        return 0
    # rem
    if id == 12:
        return 4
    # N1
    if id == 13:
        return 1
    # N2
    if id == 14:
        return 2
    # N3
    if id == 15:
        return 3


subjects = ['P396', 'P398', 'P402', 'P405', 'P406', 'P415', 'P416']
for subject_id in subjects:
    mat = sio.loadmat(f'C:\\UCLA\\{subject_id}_stages.mat')
    # format to a list
    scoring_mat = mat['stages30secSegments'].reshape(mat['stages30secSegments'].shape[0])
    scoring_py = [get_id(x) for x in scoring_mat]

    with open(f'{subject_id}_hypno.txt', 'w') as fo:
        for row in scoring_py:
            fo.write(str(row) + '\n')

    # with open(f'{subject_id}_hypno.csv', "w", newline='') as f:
    #     writer = csv.writer(f)
    #     for row in scoring_py:
    #         writer.writerow([row])
    #     # overwrite file if already exist
    #     f.truncate()

print('finish')
