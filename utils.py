import pickle
import os
import time
import re
import  sys
import numpy as np
import gensim
import json
import torch
import collections
from torch.utils.data import SequentialSampler, DataLoader


# const
NONE = 'O'
PAD = 'PAD'
convert_token = dict({'-LRB-': '(', '-RRB-': ')'})

def load_bin_vec(fname, vocab):
    """
        Loads 300x1 word vecs from Google (Mikolov) word2vec
        """
    word_vecs = np.zeros((len(vocab), 300))
    count = 0
    vocab_bin = gensim.models.KeyedVectors.load_word2vec_format(
        os.path.join(os.path.dirname(__file__), fname), binary=True)
    for word in vocab:
        if word in vocab_bin:
            count += 1
            word_vecs[vocab.index(word)]=(vocab_bin[word])
        else:
            word_vecs[vocab.index(word)] = (np.random.uniform(-0.25, 0.25, 300))
        print("found %d" %count)
    return word_vecs

def load_vocab(filename, hasPad=True):
    """Loads vocab from a file

    Args:
        filename: (string) the format of the file must be one sentence_ per line.

    Returns:
        d: dict[sentence_] = index

    """
    d = dict()
    if hasPad:
        d.update({'PAD': 0})
    with open(filename, encoding='utf-8-sig') as f:
        data = f.read().split()
        for id, w in enumerate(data):
            d[w] = len(d)

    return d

def load_trimmed_word2vec(path):
    """
    Load sentence_ embedding Word2vec from file
    :param path: path to the word2vec vectors
    :return:
        vocab: length of vocabulary
        word2id, id2word: dictionary
        word_ebeddings_matrix: contain the vector embedding of each sentence_
    """
    start = time.time()
    print('==> Loading model word2vec...')
    word2id = {}
    id2word = {}
    with open(path, 'r', encoding='utf-8') as f:
        data = f.read().split('\n')
        word_Embeddings_matrix = [[0]*int(data[0].split(' ')[1])]
        for i, line in enumerate(data[1:len(data)-1]):
            if(line ==''):
                continue
            word_vec = line.split(' ')
            word2id[word_vec[0]]= len(word2id)+1
            word_Embeddings_matrix.append([ float(val) for val in word_vec[1:]])

    id2word = dict(zip(word2id.values(),word2id.keys()))
    nwords = len(word_Embeddings_matrix)
    print('==> Finish load model ({},{})in {:.2f} sec.'.format(nwords, len(word_Embeddings_matrix[0]),time.time()-start))

    return nwords, word2id, id2word, np.array(word_Embeddings_matrix)

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)


def load_data(window, label):
    vectors = pickle.load(open("vector.bin", 'rb'))
    sents = pickle.load(open(window, 'rb'))
    anchor = pickle.load(open(label, 'rb'))
    return vectors, sents, anchor


def encode_window(tokens, anchors, entities, deps, word2id=None, event2id=None, entity2id=None, window_size=25, save=False, prefix='data/test_'):
    unk_id = word2id["UNK"]
    none_e_id = entity2id[NONE]
    pad_id = word2id[PAD]
    epad_id = entity2id[PAD]
    pospad_id = 0

    data = dict({'word_ids': [],
                 'entity_ids': [],
                 'position_ids': [],
                 'labels': []})
    count = 0
    for ids, (sent, entities_sent, deps_sent) in enumerate(zip(tokens, entities, deps)):

        for tok in np.arange(len(sent)):
            w_window, e_window, p_window = [], [], []
            if anchors[ids][tok][0] == 'I':
                count +=1
                continue

            for i in range(-window_size, window_size+1):
                if i + tok < 0 or i + tok >= len(sent):
                    w_window.append(pad_id)
                    e_window.append(epad_id)
                    p_window.append(pospad_id)

                else:
                    w_window.append(word2id.get(sent[i + tok].lower(), unk_id))
                    e_window.append(entity2id[entities_sent[i+tok]])
                    # p_window.append(abs(i)+1)
                    p_window.append(i+ window_size + 1)


            data['position_ids'].append(p_window)
            data['word_ids'].append(w_window)
            data['entity_ids'].append(e_window)
            data['labels'].append(event2id[anchors[ids][tok][2:] if anchors[ids][tok] !='O' else 'O'])


    # print(sys.getsizeof(w_windows))
    print(data['position_ids'][:3])
    print(data['word_ids'][:3])
    print(data['entity_ids'][:3])
    print(data['labels'][:3])
    print(collections.Counter(data['labels']))
    print('total inner: ', count)
    if save:
        with open(prefix + 'data.pkl', 'wb') as f:
            pickle.dump(data, f)


def load_data_json(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)
        words_sents, lab_triggers_sents, entities_sents, dep_sents = [], [], [], []
        equalToken = re.compile('==+')

        for item in data:
            words = item['words']

            golden_entities = [(range(en['head']['start'], en['head']['end']), en['entity-type']) for en in item['golden-entity-mentions']]
            entities = [NONE] * len(words)# entity parse
            for i in range(len(words)):
                for en in golden_entities:
                    if i in en[0]:
                        e_type = en[1].split(':')[-1]  # get tail of entity-type
                        if i == list(en[0])[0]:
                            e_type = 'B-' + e_type
                        else:
                            e_type = 'I-' + e_type

                        entities[i] = e_type
                        break

            deps = [] # dependency parse
            for dep in item['stanford-colcc']:
                dep = dep.split('/')
                if dep[0]!='ROOT':
                    deps.append((int(dep[-1].split('=')[1]), int(dep[-2].split('=')[1])))  # ((governor_id, depend_id),...)

            triggers = [NONE] * len(words) # trigger parse
            for ev in item['golden-event-mentions']:
                range_ = list(range(ev['trigger']['start'], ev['trigger']['end']))
                for idx_ev in range_:
                    event_type = ev['event_type'].split(':')[-1]
                    if idx_ev == range_[0]:
                        event_type = "B-" + event_type
                    else:
                        event_type = "I-" + event_type

                    triggers[idx_ev] = event_type

            for i in range(len(words)):
                if words[i] in ['-LRB-', '-RRB-']:
                    words[i] = convert_token[words[i]]
                elif equalToken.search(words[i]):
                    words[i] = '='

            words_sents.append(words)
            lab_triggers_sents.append(triggers)
            dep_sents.append(deps)
            entities_sents.append(entities)

    # print(words_sents)
    # print(lab_triggers_sents)
    # print(entities_sents)
    # print(dep_sents)
    return words_sents, lab_triggers_sents, entities_sents, dep_sents


def toconll(fpath, window_size=31, save=True, prefix='data/train'):
    words_sents, lab_triggers_sents, entities_sents, dep_sents = load_data_json(fpath)
    if save:
        with open(prefix+'conll.txt', 'a', encoding='utf-8') as f:
            for sent, labs, entities in zip(words_sents, lab_triggers_sents, entities_sents):
                for idx, (word, lab, entity ) in enumerate(zip(sent, labs, entities)):
                    if idx < len(sent)-1:
                        f.write(word+'\t'+lab+'\t'+ entity+'\n')
                    else:
                        f.write(word+'\t'+lab+'\t'+ entity+'\n\n')



def load_data_pickle(fpath, max_sent=31):
    """

    :param fpath:
    :param max_sent:
    :return:
        data: includes 5 tensors:
            - word_ids: data_size, max_sent
            - out_adjacency matrix: sparse tensor matrix: data_size, max_sent, max_sent
            - inverse adjacency matrix : inverse edge in depency graph, sparse tensor matrix:
            - entity_ids: data_size, max_sent
            - labels: data_size, max_sent

    """
    with open(fpath, 'rb') as f:
        data = pickle.load(f)

    data = [torch.LongTensor(data[feat]) for feat in data]

    return data



def checkChunk(target, predict, ori_sents, vocab_tag):
    # compare target set with predict set have printing all missing label
    correct_preds, total_correct, total_preds = 0., 0., 0.
    event_counts = []
    id2event = dict(zip(vocab_event.values(), vocab_event.keys()))

    for pred, true in zip(predict, target):
        if true != 0 and true == pred:
            correct_preds += 1
        if true != 0:
            total_correct += 1
        if pred != 0:
            total_preds += 1
            event_counts.append(id2event[pred])

    print('Event: ', collections.Counter(event_counts).most_common())
    print('\tresult: {}-{}-{}'.format(total_correct, total_preds, correct_preds))
    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

    return p * 100, r * 100, f1 * 100


def evaluate(config, eval_dataset, model, id2word, prefix=""):
    # Note that DistributedSampler samples randomly
    word2id = dict(zip(id2word.values(), id2word.keys()))
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=config.eval_batch_size)

    print("\n***** Running evaluation {} *****".format(prefix))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    ori_sent_ids = None
    model.eval()

    for batch in eval_dataloader:
        batch = tuple(t.to(config.device) for t in batch)

        with torch.no_grad():
            # input model1

            inputs = {"input_ids": batch[0],
                      "input_ners": batch[1],
                      "input_positions": batch[2],
                      "labels": batch[3]}

            outputs = model(**inputs)
            logits, tmp_eval_loss = outputs[:2]

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
            ori_sent_ids = inputs['input_ids'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
            ori_sent_ids = np.append(ori_sent_ids, inputs['input_ids'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds_ = np.argmax(preds, axis=-1)
    label_map = dict(zip(config.vocab_event.values(), config.vocab_event.keys()))
    ori_sents_list = [[word2id[word] for word in sent if word != 0] for sent in ori_sent_ids]

    prec, recall, f1 = checkChunk(out_label_ids.tolist(), preds_.tolist(), ori_sents_list, config.vocab_event)
    results = {
        "loss": eval_loss,
        "precision": prec,
        "recall": recall,
        "f1": f1,
    }
    print('\t', results)
    # for key in results.keys():
    #    print("   {} = {:.4f}".format(key, results[key]))

    return results



if __name__ == "__main__":
    nwords, word2id, id2word, _ = load_trimmed_word2vec('data/trimmed_word2vec_new.txt')
    event2id = load_vocab('../data/vocab_event.txt', False)
    entity2id = load_vocab('../data/vocab_ner_tail.txt')
    word2id.update({'PAD': 0})
    # vocab_event = event2id
    vocab_event = dict({'O' : 0})
    for key in event2id:
        if key[2:] not in vocab_event and key != "O":
            vocab_event.update({key[2:] : len(vocab_event)})
    print(vocab_event)
    print(entity2id)
    print(event2id)
    for op in ['dev','test', 'train']:
        print('-->opt: ', op)
        words_sents, lab_triggers_sents, entities_sents, dep_sents = load_data_json('data/sdata/{}.json'.format(op))
        encode_window(words_sents, lab_triggers_sents, entities_sents, dep_sents, word2id, vocab_event, entity2id, window_size=15, save=True, prefix='data/out/{}_'.format(op))
