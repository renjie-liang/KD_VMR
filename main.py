import os
import argparse
import tensorflow as tf
from tqdm import tqdm
from models.SeqPAN import SeqPAN
from models.BaseFast import BaseFast
from models.SingleTeacher import SingleTeacher
from models.MultiTeacher import MultiTeacher
from models.MultiTeacherPlus import MultiTeacherPlus

from utils.data_gen import gen_or_load_dataset
from utils.data_loader import TrainLoader, TestLoader, TrainNoSuffleLoader, TrainLoaderMultiTeacher
from utils.data_utils import load_json, save_json, load_video_features
from utils.runner_utils import eval_test_save, get_feed_dict, get_feed_dict_MultiTeacher
from utils.runner_utils import write_tf_summary, set_tf_config, eval_test, eval_test_fast
from datetime import datetime
import numpy as np
from easydict import EasyDict
from utils.utils import load_yaml
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, required=True, help='config file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument('--eval', action='store_true', help='only evaluate')
    parser.add_argument('--debug', action='store_true', help='only debug')
    parser.add_argument('--suffix', type=str, default='', help='task suffix')
    parser.add_argument('--seed', default=1234, type=int, help='random seed')
    parser.add_argument('--gpu_idx', type=str, default='0', help='indicate which gpu is used')
    return parser.parse_args()

args = parse_args()
configs = EasyDict(load_yaml(args.config))
configs['suffix'] = args.suffix


if configs.model.name.startswith("MultiTeacher"):
    feed_func = get_feed_dict_MultiTeacher
    trainloader_func = TrainLoaderMultiTeacher
else:
    feed_func = get_feed_dict
    trainloader_func = TrainLoader
    
# prepare or load dataset
dataset = gen_or_load_dataset(configs)
configs.num_chars = dataset['n_chars']
configs.num_words = dataset['n_words']

# get train and test loader
visual_features = load_video_features(configs.paths.feature_path, configs.model.max_vlen)
train_loader = trainloader_func(dataset=dataset['train_set'], visual_features=visual_features, configs=configs)
test_loader = TestLoader(datasets=dataset, visual_features=visual_features, configs=configs)
# train_nosuffle_loader = TrainNoSuffleLoader(datasets=dataset['train_set'], visual_features=visual_features, configs=configs)

model_dir = 'ckpt/{}'.format(configs.task)
logs_dir = 'logs/{}'.format(configs.task)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)
date_str = datetime.now().strftime("%Y%m%d_%H%M%S") 
log_path = os.path.join(logs_dir, date_str + "_{}.txt".format(configs.model.name))
             
if not args.eval:
    eval_period = train_loader.num_batches()
    # save_json(vars(configs), filename=os.path.join(model_dir, "configs.json"), save_pretty=True)
    with tf.Graph().as_default() as graph:
        model = eval(configs.model.name)(configs=configs, graph=graph, word_vectors=dataset['word_vector'])
        sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        best_r1i7, best_miou = -1.0, -1.0
        best_r1i7_performance, best_miou_performance = None, None
        with tf.compat.v1.Session(config=sess_config) as sess:
            saver = tf.compat.v1.train.Saver(max_to_keep=3)
            sess.run(tf.compat.v1.global_variables_initializer())
            score_writer = open(log_path, mode="w", encoding="utf-8")
            score_writer.write(json.dumps(configs))
            for epoch in range(configs.train.epochs):
                if epoch < 2:
                    cur_lr = 0.000001
                else:
                    cur_lr = configs.train.lr * (1.0 - epoch / configs.train.epochs)

                for data in tqdm(train_loader.batch_iter(), total=train_loader.num_batches(),
                                 desc='Epoch %d / %d' % (epoch + 1, configs.train.epochs)):
                    feed_dict = feed_func(data, model, lr=cur_lr, drop_rate=configs.model.droprate, mode='train')

                    # _, tmp = sess.run([model.train_op, model.tmp], feed_dict=feed_dict)
                    # print(tmp)
                    # _, tmp, tmp0, tmp1, tmp2 = sess.run([model.train_op, model.tmp, model.tmp0, model.tmp1, model.tmp2], feed_dict=feed_dict)
                    # # print(tmp.shape)
                    # print(tmp.min(), tmp.max())
                    # print(tmp0.min(), tmp0.max())
                    # print(tmp1.min(), tmp1.max())
                    # print(tmp2.min(), tmp2.max())

                    _, loss,  global_step = sess.run([model.train_op, model.loss, model.global_step], feed_dict=feed_dict)

                    if global_step % eval_period == 0:  # evaluation
                        r1i3, r1i5, r1i7, mi, value_pairs, score_str = eval_test(
                            sess=sess, model=model, data_loader=test_loader, epoch=epoch + 1, global_step=global_step, prefix="")
                        score_writer.write(score_str)
                        score_writer.flush()
                        print(score_str, flush=True)

                        # r1i3, r1i5, r1i7, mi, value_pairs, score_str = eval_test_fast(
                        #     sess=sess, model=model, data_loader=test_loader, epoch=epoch + 1, global_step=global_step, prefix="fast")
                        # score_writer.write(score_str)
                        # score_writer.flush()
                        # print('\nFast TEST Epoch: %2d | Step: %5d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f | mIoU: %.2f' % (
                        #     epoch + 1, global_step, r1i3, r1i5, r1i7, mi), flush=True)


                       ## save the model according to the result of Rank@1, IoU=0.7
                        if r1i7 > best_r1i7:
                            best_r1i7 = r1i7
                            filename = os.path.join(model_dir, "best_{}.ckpt".format(configs.model.name))
                            saver.save(sess, filename)
                            best_r1i7_performance = score_str
                        if mi > best_miou:
                            best_miou = mi
                            best_miou_performance = score_str
                            
            score_writer.write("\n\nOptimal performance at the highest R1i7 epoch and the highest mIoU epoch.\n")           
            score_writer.write(best_r1i7_performance)
            score_writer.write(best_miou_performance)
            score_writer.close()

else:
    print(model_dir)
    if not os.path.exists(model_dir):
        raise ValueError('no pre-trained model exists!!!')
    pre_configs = load_json(os.path.join(model_dir, "configs.json"))
    parser.set_defaults(**pre_configs)
    configs = parser.parse_args()
    # load model and test
    with tf.Graph().as_default() as graph:
        model = SeqPAN(configs=configs, graph=graph, word_vectors=dataset['word_vector'])
        sess_config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=sess_config) as sess:
            saver = tf.compat.v1.train.Saver()
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            r1i3, r1i5, r1i7, mi, *_ = eval_test(sess=sess, model=model, data_loader=test_loader, mode=configs.mode)
            r1i3, r1i5, r1i7, mi, *_ = eval_test_fast(sess=sess, model=model, data_loader=train_nosuffle_loader, task=configs.task, suffix=configs.suffix, mode=configs.mode)
            r1i3, r1i5, r1i7, mi, *_ = eval_test_save(sess=sess, model=model, data_loader=train_nosuffle_loader, task=configs.task, suffix=configs.suffix, mode=configs.mode)
            print("\n" + "\x1b[1;31m" + "Rank@1, IoU=0.3:\t{:.2f}".format(r1i3) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "Rank@1, IoU=0.5:\t{:.2f}".format(r1i5) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "Rank@1, IoU=0.7:\t{:.2f}".format(r1i7) + "\x1b[0m", flush=True)
            print("\x1b[1;31m" + "{}:\t{:.2f}".format("mean IoU".ljust(15), mi) + "\x1b[0m", flush=True)
