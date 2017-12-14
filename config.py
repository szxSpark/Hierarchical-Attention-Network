import pickle

class Config:
    has_cuda = True
    is_training = True
    is_pretrain = True
    force_word2index = False
    embedding_path = "./word2vec/pretrain_emb.alltrain.256d.npy"
#    embedding_path = "./word2vec/pretrain_emb.128d.npy"
    test_path = './corpus/seg_test.txt'
#    test_path = './corpus/test_preprocessed.txt'
    result_path = './results/test_result.json'
    data_path = './corpus/seg_train.txt'
#    data_path = './corpus/train_m_preprocessed.txt'
    model_path = './pickles/params.pkl'
    
    index2word_path = './pickles/index2word.all.pkl'
    word2index_path = './pickles/word2index.all.pkl'
    model_names = ['fastText',
                   'TextCNN', 
                   'TextRCNN',
                   'TextRNN',
                   'HAN',
                   'CNNWithDoc2Vec',
                   'RCNNWithDoc2Vec',
                   'CNNInception'
                  ]

    batch_size = 32 # 64 if has cuda
    step = 6000//batch_size     # 3000 // batch_size if has cuda
    num_workers = 1
    vocab_size = 0
    min_count = 5
    max_text_len = 2000
    embedding_size = 256
    num_class = 8
    learning_rate = 0.001
    if not is_pretrain:
        learning_rate2 = 0.001  
    else:
        learning_rate2 = 0.0    # 0.0 if pre train emb
    lr_decay = 0.75
    begin_epoch = 2
    weight_decay = 0.0
    dropout_rate = 0.5
    epoch_num = 6
    epoch_step = max(1, epoch_num // 20)

    # HAN
    han_batch_size = 32
    num_sentences = 55	# 20
    sequence_length = 70 
    word_hidden_size =  50 
    sentence_hidden_size = 50
    word_context_size = 100
    sentence_context_size = 100

    loss_weight_value = [ 
         0.4243,
         0.5050,
         0.8118,
         0.9436,
         0.7862,
         0.6290,
         0.2412,
         0.8248,
    ]

