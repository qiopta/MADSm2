import transformers

DEVICE = "cuda"
MAX_LEN = 64
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "/home/azureuser/cloudfiles/code/Users/qiopta26/m2/bert-ptd/input/pytorch_model.bin" #edited to /bert-ptd/folder
TRAINING_FILE = "/home/azureuser/cloudfiles/code/Users/qiopta26/m2/bert-ptd/input/WikiLarge_Train.csv" #edited to /bert-ptd/ folder and WikiLarge_Train.csv
TEST_FILE = "/home/azureuser/cloudfiles/code/Users/qiopta26/m2/bert-ptd/input/WikiLarge_Test.csv" #edited to /bert-ptd/ folder and WikiLarge_Train.csv
TRAINED_MODEL = "/home/azureuser/cloudfiles/code/Users/qiopta26/m2/bert-ptd/input/pytorch_model.bin" #added for pred_sumbit.py to use for inference
TRAIN_LOG =  "/home/azureuser/cloudfiles/code/Users/qiopta26/m2/bert-ptd/input/train_log.txt" #added for pred_sumbit.py to use for inference
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
