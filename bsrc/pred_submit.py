import torch
import config
import dataset 
import pandas as pd
import numpy as np
import kaggle
from model import BERTBaseUncased
import engine

def create_preds():
    ''' 
    A function that gets the test data and pre_trained pytorch model
    runs inference only and creates an output, 
    this will need submission directly into Kaggle via command line API
    NB; LINE 35 in dataset.py ('Targets' etc) needs to be #'d out for this to run. 
    '''

    dfx = pd.read_csv(config.TEST_FILE).fillna("none")
    df_test = dfx.reset_index(drop=True) #loads CSV file

    test_dataset = dataset.BERTDataset(
        review=df_test.original_text.values, target=df_test.label.values
    ) #pre-procssing text for BERT use

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )
    
    device = torch.device(config.DEVICE)
    model = BERTBaseUncased()
    model.load_state_dict(torch.load(config.TRAINED_MODEL)) #load best model output from training
    model.to(device)

    outputs = engine.inference_fn(test_data_loader, model, device) #removed , targets
    outputs = np.array(outputs) >= 0.5
    outputs = outputs.astype(int)

    #next turn outputs array into CSV for uploading into kaggle
    pred_df = pd.DataFrame(data=outputs).reset_index()
    pred_df.columns = ['id', 'label']
    pred_df.to_csv('preds.csv', index=False)

    #make sumbission to Kaggle
    #print ("Making Kaggle submission...")
    #!kaggle competitions submit umich-siads-695-predicting-text-difficulty -f preds.csv -m "Testing submission via Azure ML engine, based on 1 EPOCH Bert Model est. 75%"

if __name__ == "__main__":
    create_preds()

