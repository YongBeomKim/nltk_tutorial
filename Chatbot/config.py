train_dir = "./model"           # model directory
log_dir   = "./logs"            # log directory
ckpt_name = "conversation.ckpt" # checkpoint

BATCH_SIZE = 100
EPOCH      = 2000

n_hidden = 128
n_layer  = 3
learning_rate  = 0.001
max_decode_len = 20

DATA_PATH = "./data/conversation2.txt" #data
VOC_PATH  = "./data/conversation.voc"  #data.voc

# Special tokens
PAD = "_PAD_"
GO  = "_GO_"
EOS = "_EOS_"
UNK = "_UNK_"

PAD_ID = 0
GO_ID  = 1
EOS_ID = 2
UNK_ID = 3
ALL = [PAD_ID, GO_ID, EOS_ID, UNK_ID]