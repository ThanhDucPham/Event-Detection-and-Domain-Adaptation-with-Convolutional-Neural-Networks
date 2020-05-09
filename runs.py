from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, RandomSampler
from tqdm import tqdm_notebook, tqdm
import math
import os
import random
import collections
import numpy as np
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from utils import *
from model import *
from sklearn.utils.class_weight import compute_class_weight

if __name__ == "__main__":

    print('--> Load vocab: ')
    word2id = load_vocab('data/vocab_word.txt')
    event2id = load_vocab('data/vocab_event.txt', False)
    entity2id = load_vocab('data/vocab_ner_tail.txt')
    nwords, word2id, id2word, pretrained_embeddings = load_trimmed_word2vec('data/trimmed_word2vec_new.txt')
    # print('Data preparation')
    # word2id.update({'PAD': 0})
    # event2id.update({'PAD': -100})
    # vocab_event = event2id
    # # vocab_event = dict({'O' : 0})
    # # for key in event2id:
    # #     if key[2:] not in vocab_event and key[2:] != '':
    # #         vocab_event.update({key[2:] : len(vocab_event)})
    # # print(vocab_event)
    # for op in ['dev', 'test', 'train']:
    #     print('-->opt: ', op)
    #     words_sents, lab_triggers_sents, entities_sents, dep_sents = load_data_json('data/{}.json'.format(op))
    #     encode_window2(words_sents, lab_triggers_sents, entities_sents, dep_sents, word2id, vocab_event, entity2id,
    #                    window_size=31, save=False, prefix='data/loaddata/{}_'.format(op))

    print('-> Load data')
    train_data = load_data_pickle('data/out/train_data.pkl', max_sent=31)
    dev_data = load_data_pickle('data/out/dev_data.pkl', max_sent=31)
    test_data = load_data_pickle('data/out/test_data.pkl', max_sent=31)

    train_dataset = TensorDataset(*train_data)
    dev_dataset = TensorDataset(*dev_data)
    test_dataset = TensorDataset(*test_data)

    print('input_ids shape: ', train_data[0].shape)
    print('adj_out matrix shape: ', train_data[1].shape)



    print('-> Build model')
    config = Config()
    config.set_seed(150)
    config.nstep_logging = 4900
    config.eval_batch_size = 128
    config.batch_size = 50
    config.learning_rate = 5e-3
    config.num_epoch = 300
    config.warmup_steps = 4000
    config.window_size = 15
    config.fine_tune = True
    config.change_lr_steps = 4000

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    print(collections.Counter(train_data[3].numpy()))
    print(list(range(len(config.vocab_event))))
    weightsLoss_classes = compute_class_weight('balanced', classes=list(range(len(config.vocab_event))),
                                               y=train_data[3].numpy())

    model = CNNModel(config,
                     class_weights=torch.from_numpy(weightsLoss_classes).type(torch.float32).to(config.device),
                     pretrained_embeddings=torch.tensor(pretrained_embeddings, dtype=torch.float32))
    model.to(config.device)

    optimizer = optim.Adadelta(model.params_requires_grad(),
                               weight_decay=config.weight_decay,
                               lr=config.learning_rate,
                               eps=config.adam_eps)
    print(model)

    global_step = 0.
    f1_best = 0.
    f1_test = 0.
    logging_loss, tr_loss = 0., 0.
    epoch_improve = 0.
    restart_used = 0
    model_name = 'model_gcn_2018.ckpt'
    log_name = 'log_gcn2018.txt'
    tensorboard_name = 'model_1.ckpt'
    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config.batch_size)
    total_steps = int(len(train_loader) / config.change_lr_steps * config.num_epoch) + 1
    config.warmup_steps = total_steps // 5
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps,
                                                num_training_steps=total_steps)
    tb_writer = SummaryWriter(os.path.join(config.output_dir, tensorboard_name))
    identity_matrix = torch.eye(config.max_sent).unsqueeze(0)

    print('-> Start training process')
    print('nepoch: ', config.num_epoch)
    print('step per epoch: ', len(train_loader))
    print('total step change learning rate: ', total_steps)
    print('warm up steps: ', config.warmup_steps)

    for ep in range(config.num_epoch):
        train_iterator = tqdm(train_loader)
        for step, batch in enumerate(train_iterator):
            global_step += 1
            model.train()
            model.zero_grad()
            batch = tuple(t.to(config.device) for t in batch)
            identity_matrix_batch = identity_matrix.repeat(batch[1].shape[0], 1, 1).to(config.device)

            inputs = {"input_ids": batch[0],
                      "input_ners": batch[1],
                      "input_positions": batch[2],
                      "labels": batch[3]}

            _, loss = model(**inputs)
            tr_loss += loss.item()
            loss.backward()
            # train_iterator.set_description("Epoch {}/{}(lr = {:.10f})-l={:.3f}".format(int(ep), int(config.num_epoch), optimizer.param_groups[0]['lr'], loss.item()))
            torch.nn.utils.clip_grad_norm_(model.params_requires_grad(), 1)
            optimizer.step()
            if global_step % config.change_lr_steps == 0:
                scheduler.step()

            if config.nstep_logging > 0 and global_step % config.nstep_logging == 0: # or step == len(train_iterator) - 1):
                print('lr = {}\n'.format(optimizer.param_groups[0]['lr']))
                # print(loss)

                # print('check', ep, global_steps)
                results = evaluate(config, dev_dataset, model, word2id,
                                   prefix='dev set, step {}/{}'.format(global_step, ep))
                test_results = evaluate(config, test_dataset, model, word2id,
                                        prefix='test set, step {}/{}'.format(global_step, ep))

                for key, value in results.items():
                    if key != 'loss':
                        tb_writer.add_scalar("{} score".format(key), value, global_step)
                tb_writer.add_scalar("learning rate", scheduler.get_last_lr()[0], global_step)
                tb_writer.add_scalars("loss", {'train_loss': (tr_loss - logging_loss) / config.nstep_logging,
                                               'dev_loss': results['loss']}, global_step)

                logging_loss = tr_loss


                if test_results['f1'] > f1_test:
                    f1_test = test_results['f1']
                    print('-->Test new best score! f1_test = ', f1_test)
                if results['f1'] > f1_best:
                    f1_best = results['f1']
                    epoch_improve = ep
                    print('--> New best score! f1 = ', f1_best)
                    torch.save(model.state_dict(), os.path.join(config.output_dir, model_name))
                    with open(os.path.join(config.output_dir, log_name), 'a', encoding='utf-8') as f:
                        f.write('Epoch: {:3.0f}, step: {:4.0f} global_step: {:5.0f} (lr= {:.7f})\n\
                                                Results: P= {:.4f} - R= {:.4f} - F= {:.4f} \n \t--=>>>New best score!\n'.format(
                            ep, step, global_step, optimizer.param_groups[0]['lr'],
                            results['precision'],
                            results['recall'],
                            results['f1']))


                else:
                    with open(os.path.join(config.output_dir, log_name), 'a', encoding='utf-8') as f:
                        f.write('Epoch: {:3.0f}, step: {:4.0f} global_step: {:5.0f} (lr= {:.7f})\n\
                                Results: P= {:.4f} - R= {:.4f} - F= {:.4f}\n'.format(ep, step, global_step,
                                                                                     optimizer.param_groups[0]['lr'],
                                                                                     results['precision'],
                                                                                     results['recall'],
                                                                                     results['f1']))

                    if ep - epoch_improve > 10 and f1_best > 10: # start try to reload model when score model reach a special threshold
                        if restart_used > config.max_restart:
                            print('Restarting model is run out')
                            break
                        else:
                            restart_used += 1
                            print('--->>>RELOAD MODEL from epoch {}'.format(epoch_improve))
                            with open(os.path.join(config.output_dir, log_name), 'a', encoding='utf-8') as f:
                                f.write('---->>>>RELOAD MODEL FROM EPOCH {}\n'.format(epoch_improve))
                            model.load_state_dict(torch.load(os.path.join(config.output_dir, model_name)))
                            epoch_improve = ep
                            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                                        num_warmup_steps=config.warmup_steps * ( total_steps - global_step) / total_steps,
                                                                        num_training_steps=total_steps - global_step)

        # if (ep+1) % 5 == 0:
        #     output.clear()

    print('-->FINAL TEST')
    test_results = evaluate(config, test_dataset, model, word2id, prefix='test set- final test')





