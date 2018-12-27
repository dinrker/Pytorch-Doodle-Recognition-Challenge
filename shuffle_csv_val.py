import argparse
import ast
import os
import datetime as dt
from tqdm import tqdm
import pandas as pd

# Example:
# python shuffle_csv_val.py --val_path data_2018-11-24-22-49/val.csv

# python shuffle_csv_val.py --val_path data_val/val.csv --nrows 20000
# python shuffle_csv_val.py --nrows 50000
    
parser = argparse.ArgumentParser(description='Generate new csv files')
parser.add_argument('--root', default='/home/jun/quick-draw/input/csv/train_simplified', type=str, metavar='R',
                    help='path to simplified csvs')
parser.add_argument('--ncsvs', default=100, type=int, metavar='C',
                    help='number of new csv files')
parser.add_argument('--nrows', default=50000, type=int, metavar='R',
                    help='number of samples in each category')
parser.add_argument('--cols', default=None, type=str, metavar='C',
                    help='extra columns to be saved in dataset')
parser.add_argument('--cats', default=None, type=str, metavar='C',
                    help='selected categories to be saved in dataset')
parser.add_argument('--val', default=False, type=str, metavar='V',
                    help='generate validation data')
parser.add_argument('--val_num', default=200, type=int, metavar='VN',
                    help='number of validation samples in each category')
parser.add_argument('--val_path', default='input_data/data_val/val.csv', type=str, metavar='VP',
                    help='validation data path')


def f2cat(filename: str) -> str:
    return filename.split('.')[0]

class Simplified():
    def __init__(self, input_path, prob_path=None, val_path=None):
        self.input_path = input_path
        self.prob_path = prob_path
        self.val_path = val_path

    def list_all_categories(self):
        files = os.listdir(self.input_path)
        return sorted([f2cat(f) for f in files], key=str.lower)

    def read_training_csv(self, category, nrows=None, usecols=None, drawing_transform=False):
        df = pd.read_csv(os.path.join(self.input_path, category+'.csv'),parse_dates=['timestamp'],usecols=usecols)
        # df = df[df['recognized']==True]
        if self.prob_path is not None:
            prob = pd.read_csv(os.path.join(self.prob_path, 'prob_'+category+'.csv'))
            df['prob'] = prob['prob']
            df = df[df['prob']>0.1]
        if self.val_path is not None:
            val = pd.read_csv(self.val_path)
            val_ids = set(list(val['key_id']))
            df['in_val'] = df['key_id'].apply(lambda x: x in val_ids)
            df = df[df['in_val'] == False]
        df = df.sample(n=nrows)
        if drawing_transform:
            df['drawing'] = df['drawing'].apply(ast.literal_eval)
        return df
 

# def new_train(root, NROWS = 50000, cols = None, cats = None, prob_path = None, val_path = None):
def new_train(root, NROWS = 25000, cols = None, cats = None, prob_path = None, val_path = None):

    start = dt.datetime.now()
    s = Simplified(root, val_path=val_path)
    if cols is None:
        columns = []
    else:
        columns = list(map(str, cols.replace(', ',',').split(',')))
    
    if cats is None:
        categories = s.list_all_categories()
    else:
        categories = list(map(str, cats.replace(', ',',').split(',')))
    kk = int(NROWS / 1000) 
    folder = 'data_'+start.strftime("%Y-%m-%d-%H-%M") + '_%sk' % str(kk)
    os.mkdir(folder)
    collist = ['key_id', 'drawing', 'y'] + columns
    df = pd.DataFrame(columns = collist)
    df.to_csv(folder+'/train.csv', index=False)
    
    for y, cat in tqdm(enumerate(categories)):
        df = s.read_training_csv(cat, nrows=NROWS)
        df['y'] = y
        df = df[collist]
        df.to_csv(folder+'/train.csv', mode='a', header=False, index=False)

        
def new_val(root, NROWS = 200, cols = None, cats = None, prob_path = None):
    
    start = dt.datetime.now()
    s = Simplified(root)
    if cols is None:
        columns = []
    else:
        columns = list(map(str, cols.replace(', ',',').split(',')))
    
    if cats is None:
        categories = s.list_all_categories()
    else:
        categories = list(map(str, cats.replace(', ',',').split(',')))
    folder = 'data_'+start.strftime("%Y-%m-%d-%H-%M")
    os.mkdir(folder)
    collist = ['key_id', 'drawing', 'y'] + columns
    df = pd.DataFrame(columns = collist)
    df.to_csv(folder+'/val.csv', index=False)
    
    for y, cat in tqdm(enumerate(categories)):
        df = s.read_training_csv(cat, nrows=NROWS)
        df['y'] = y
        df = df[collist]
        df.to_csv(folder+'/val.csv', mode='a', header=False, index=False)

                
def main():
    args = parser.parse_args()
    if args.val:
        new_val(args.root, NROWS = args.val_num, cols=args.cols, cats=args.cats)
    else:
        new_train(root=args.root, NROWS=args.nrows, cols=args.cols, cats=args.cats, val_path=args.val_path)
    

if __name__ == '__main__':
    main()