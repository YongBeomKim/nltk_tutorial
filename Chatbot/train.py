import os, random, math
#os.environ['CUDA_VISIBLE_DEVICES']='1'
import tensorflow as tf
import config
from model import sequence2sequence
from Conversation import Conversation

checkpoint = tf.train.get_checkpoint_state(config.train_dir)
checkpoint_path = os.path.join(config.train_dir,config.ckpt_name)

def train(conv, batch_size,epoch):

    model = sequence2sequence(conv.voc_size)

    with tf.Session() as sess:

        if checkpoint and tf.train.checkpoint_exists(checkpoint.model_checkpoint_path):
            model.saver.restore(sess,checkpoint.model_checkpoint_path)
        else:
            print("Building a model")
            sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(config.log_dir,sess.graph)
        total_batch = int(math.ceil(len(conv.conversation)/float(batch_size)))

        for step in range(total_batch * epoch):
            enc_input, dec_input,dec_target=conv.next_batch(batch_size)

            _,loss = model.train(sess,enc_input,dec_input,dec_target)

            if(step+1)% 50 == 0:
                model.logs(sess, writer, enc_input, dec_input, dec_target)
                model.saver.save(sess, checkpoint_path, global_step = model.global_step)
                print('Step:', '%06d' % model.global_step.eval(),'cost =', '{:.6f}'.format(loss))

        model.saver.save(sess, checkpoint_path, global_step = model.global_step)
        print("Finished")


def main():

    conv = Conversation()
    conv.Load_voc(config.VOC_PATH)
    conv.Load_conversation(config.DATA_PATH)

    print("size",conv.voc_size)
    train(conv, batch_size = config.BATCH_SIZE, epoch = config.EPOCH)


if __name__=="__main__":
    main()
