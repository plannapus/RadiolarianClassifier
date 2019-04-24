import os
dname = os.path.dirname(os.path.abspath(__file__))
os.chdir(dname)
import code.evaluate as evaluate
import code.train as train
import subprocess
from numpy.random import choice, seed
import re
import shutil

seed(2018)
serie = "102"
pictures = []
#Collect all pictures:
for dirpath, dirnames, filenames in os.walk("datasets/dataset%s" % (serie,)):
    for filename in filenames:
        if "jpg" in filename:
            pic = os.path.join(dirpath, filename)
            species = os.path.basename(dirpath)
            spl = filename.split()
            del spl[2]
            spl = " ".join(spl)
            spl = re.sub(".jpg","",spl)
            specimen = re.sub('[a-zA-Z]$','',spl)
            pictures.append({'species': species, 'path': pic, 'specimen': specimen})

#Split each species into 10 groups of specimens
species = set([k['species'] for k in pictures])
for i in species:
    specimens = set([k['specimen'] for k in pictures if k['species']==i])
    group = choice(range(10),len(specimens))
    while len(set(group))<10:
        group = choice(range(10),len(specimens))
    for j in range(len(specimens)):
        index = [k for k, v in enumerate(pictures) if v['specimen'] == list(specimens)[j]]
        for ind in index:
            pictures[ind]['group'] = group[j]

for i in range(10):
    paths = {}
    r = range(10)
    del r[i]
    validation = choice(r, 1) #Take another group randomly as the validation group
    global_testing_set = [k['path'] for k in pictures if k['group'] == i]
    for j in species:
        sp = [k for k in pictures if k['species'] == j]
        training_set = [os.path.basename(k['path']) for k in sp if k['group'] != i and k['group'] != validation[0]]
        testing_set = [os.path.basename(k['path']) for k in sp if k['group'] == i]
        validation_set = [os.path.basename(k['path']) for k in sp if k['group'] == validation[0]]
        paths[j] = {
            'dir': os.path.basename(os.path.dirname([k['path'] for k in sp][0])),
            'training': training_set,
            'testing': testing_set,
            'validation': validation_set,
        } #Building the dictionary containing training, validation and testing sets for each species
    model_file = "./models/%s.pb" % (serie,)
    dataset = "./test_sets/%s" % (serie,)
    label_file = "./labels/%s.txt" % (serie,)
    if os.path.exists(dataset): #Erase folder if exists
        shutil.rmtree(dataset)
    os.mkdir(dataset) #Recreate it
    for file in global_testing_set:
        shutil.copy(file, dataset) #Populate it with the actual testing set of that iteration
    training = train.tenfold(seed=2018, architecture="mobilenet_1.0_224", paths= paths,
                            image_dir="./datasets/dataset%s" % (serie,), output_graph=model_file,
                            output_labels=label_file, model_dir='tmp/models/',
                            bottleneck_dir='tmp/%s/bottlenecks/'%(serie,),
                            how_many_training_steps=4000, learning_rate=0.01, print_misclassified_test_images=True)
    save_file = "result_%s_%d.txt" % (serie,i)
    evaluate.evaluateDirectory(model_file,dataset,label_file,1,save_file)
