import nltk.corpus.reader.sentiwordnet as snw
SWN_FILENAME = "SentiWordNet/SentiWordNet_3.0.0_20130122.txt"
sw = snw.SentiWordNetCorpusReader('', [SWN_FILENAME])

def get_score_word(word, pos_tag):

    global_pos = 0
    global_neg = 0

    senses = list(sw.senti_synsets(word, pos_tag))

    for i in range(len(senses)):
        sens = list(senses)[i]
        pos_score = sens.pos_score()
        neg_score = sens.neg_score()

        global_pos += pos_score
        global_neg += neg_score


    return global_pos, global_neg