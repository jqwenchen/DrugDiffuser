import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from img2mol.inference import *
from tqdm import tqdm
import pandas as pd


img2mol = Img2MolInference(model_ckpt="model/model.ckpt", local_cddd=True)

def runlivertox():
    livertox_dataset = [i for i in os.listdir('../drug-diffuser-0717/datasets/LiverTox/') if i.endswith('.jpg')]
    livertox_result = []
    for image in tqdm(livertox_dataset):
        tmp_res = img2mol(filepath=f'../drug-diffuser-0717/datasets/LiverTox/{image}')
        livertox_result.append(tmp_res)
    df_livertox = pd.DataFrame(livertox_result)
    df_livertox.iloc[:, :3].to_csv('livertox_result.csv', index=False)

def runLactMed():
    LactMed_dataset = [i for i in os.listdir('../drug-diffuser-0717/datasets/LactMed/') if i.endswith('.jpg')]
    LactMed_result = []
    for image in tqdm(LactMed_dataset):
        tmp_res = img2mol(filepath=f'../drug-diffuser-0717/datasets/LactMed/{image}')
        LactMed_result.append(tmp_res)
        if (len(LactMed_result) + 1) % 50 == 0:
            print(len(LactMed_result))
    df_lac = pd.DataFrame(LactMed_result)
    df_lac.iloc[:, :3].to_csv('LactMed_result.csv', index=False)

if __name__ == '__main__':
    runlivertox()