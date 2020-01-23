PATHS = {
    # IMAGE/MESH GENERATION
    'index': 'UM100/index.csv',
    'data': 'UM100/src/data/',
    'mesh': ['UM100/src/spinemesh@jason/', 'UM100/src/spinemesh@liang/'],
    'ext': ['jason', 'liang'],
    'pair': ['UM100/ImageMeshPair/image/', 'UM100/ImageMeshPair/point/'],
    'r_img': "C:/Research/LumbarSpine/Github/generative-model-robustness/dataset/UM100/ImageMeshPair/image/",
    'r_msh': "C:/Research/LumbarSpine/Github/generative-model-robustness/dataset/UM100/ImageMeshPair/point/"
}

TRAIN = ['Arntzen', 'Askew', 'Athanas', 'Ausman', 'Ballentine', 'Barrera',
         'Belforte', 'Bennett', 'Bickford', 'Bolt', 'Boucher', 'Carlisle',
         'Cheville', 'Claude', 'Crocker', 'Crowe', 'Dana', 'Davila',
         'Dawkins', 'Dehart', 'Deuel', 'Diamond', 'DICOM', 'Elliott',
         'Farry', 'Federle2', 'Fuentes', 'Galang', 'Garverick', 'Gatt',
         'Gayman', 'Gifford', 'Gilpin', 'Grinstein', 'Harms', 'Harton',
         'Heins', 'Hellums', 'Henson', 'Hewitt', 'Hindman', 'Houston',
         'Karlin', 'Kinney', 'Kotsonis', 'Lam', 'Laware', 'Lebron',
         'Litvak', 'Meade', 'Melton', 'Minkoff', 'Murray', 'Nassar',
         'Nederveld', 'Paige', 'Paul', 'Perrino', 'Peters', 'Player',
         'Popish', 'Pound', 'Reiter', 'Roy', 'Rupert', 'Scales', 'Schafer',
         'Sheldon', 'Shupack', 'Sievert', 'Spear', 'Spruell', 'Stacy',
         'Swint', 'Symonds', 'Talwar', 'Tolle', 'Triggs', 'Wake', 'Weien']

TEST = ['Allison', 'Baker', 'Budd', 'Derr', 'Epp', 'Etezadi', 'Federle',
        'Fortune', 'Gamse', 'Garza', 'Kastensmidt', 'Keleher', 'Lacy',
        'Oliver', 'Roth', 'Ryan', 'Schild', 'Shidle', 'Turner']

# Stewart data lost, then removed from TEST SET