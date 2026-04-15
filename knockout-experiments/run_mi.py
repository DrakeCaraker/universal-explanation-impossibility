import sys,os,csv,urllib.request,zipfile
sys.path.insert(0,'.')
if not os.path.exists('SST-2/train.tsv'):
    urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/glue/data/SST-2.zip','SST-2.zip')
    zipfile.ZipFile('SST-2.zip','r').extractall('.')
import mech_interp_rashomon_gpu as mi
def load_sst2_tsv(n_train=5000,n_val=500,n_test=200):
    train_texts,train_labels=[],[]
    with open('SST-2/train.tsv','r') as f:
        reader=csv.DictReader(f,delimiter='\t')
        for row in reader:
            train_texts.append(row['sentence'])
            train_labels.append(int(row['label']))
    test_texts,test_labels=[],[]
    with open('SST-2/dev.tsv','r') as f:
        reader=csv.DictReader(f,delimiter='\t')
        for row in reader:
            test_texts.append(row['sentence'])
            test_labels.append(int(row['label']))
    vs=min(n_train,len(train_texts)-n_val)
    vt=train_texts[vs:vs+n_val]
    vl=train_labels[vs:vs+n_val]
    print(f'  SST-2: {min(n_train,len(train_texts))} train, {len(vt)} val, {min(n_test,len(test_texts))} test')
    return train_texts[:n_train],train_labels[:n_train],vt,vl,test_texts[:n_test],test_labels[:n_test]
mi.load_sst2=load_sst2_tsv
mi.main()
