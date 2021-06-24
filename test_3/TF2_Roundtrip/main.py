from __future__ import division
import numpy as np
import tf2lib as tl
import os
import matplotlib.pyplot as plt
import tqdm
import argparse
import imlib as im
import tensorflow as tf
from tensorflow import keras
import model
import util
import functools

'''
Instructions: Roundtrip model for conditional density estimation (e.g., images)
    x,y - data drawn from base density and observation data (target density)
    y_  - learned distribution by G(.), namely y_=G(x)
    x_  - learned distribution by H(.), namely x_=H(y)
    y__ - reconstructed distribution, y__ = G(H(y))
    x__ - reconstructed distribution, x__ = H(G(y))
    G(.)  - generator network for mapping x space to y space
    H(.)  - generator network for mapping y space to x space
    Dx(.) - discriminator network in x space (latent space)
    Dy(.) - discriminator network in y space (observation space)
'''
# ==============================================================================
# =                                   param                                    =
# ==============================================================================

parser = argparse.ArgumentParser('')
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--dx', type=int, default=100)
parser.add_argument('--dy', type=int, default=784)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--epoch_decay', type=int, default=100)
parser.add_argument('--alpha', type=float, default=10.0)
parser.add_argument('--beta', type=float, default=10.0)
parser.add_argument('--timestamp', type=str, default='')
parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--df', type=float, default=1, help='degree of freedom of student t distribution')
args = parser.parse_args()
x_dim = args.dx
y_dim = args.dy
batch_size = args.bs
epochs = args.epochs


alpha = args.alpha
beta = args.beta
df = args.df
timestamp = args.timestamp

is_train = args.train
# ==============================================================================
# =                                    data                                    =
# ==============================================================================

dataset,train_label,len_dataset = util.make_28x28_dataset(batch_size)
A2B_pool = util.ItemPool(50)
B2A_pool = util.ItemPool(50)
# ==============================================================================
# =                                   models                                   =
# ==============================================================================
g_net = model.Generator_img(input_dim=x_dim, output_dim=y_dim, name='g_net', nb_layers=3, nb_units=256)
h_net = model.Encoder_img(input_dim=y_dim, output_dim=x_dim, name='h_net', nb_layers=3, nb_units=256, cond=True)
dx_net = model.Discriminator(input_dim=x_dim, name='dx_net', nb_layers=3, nb_units=128)
dy_net = model.Discriminator_img(input_dim=y_dim, name='dy_net', nb_layers=3, nb_units=128)
d_loss_fn, g_loss_fn = util.get_lsgan_losses_fn()
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()
G_lr_scheduler = util.LinearDecay(0.0002, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = util.LinearDecay(0.0002, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=0.5,beta_2=0.9)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=0.5,beta_2=0.9)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

@tf.function
def train_G(A, B):
    with tf.GradientTape() as t:
        # y_ = g_net(tf.concat([A , nb_classes],axis=1), training=True)  # G()
        y_ = g_net(A, training=True)  # G()
        # J = batch_jacobian(y_, A)
        x_ = h_net(B, training=True)  # H()
        x__ = h_net(y_, training=True)  # H(G())
        # x_combine_ = tf.concat([x_ , nb_classes],axis=1)
        # y__ = g_net(x_combine_, training=True)  # G(H())
        y__ = g_net(x_, training=True)  # G(H())
        # dy_ = dy_net(tf.concat([y_ , nb_classes],axis=1), training=True)  # D(G())
        dy_ = dy_net(y_, training=True)  # D(G())
        dx_ = dx_net(x_, training=True)  # D(H())
        g_loss_adv = g_loss_fn(dy_)
        h_loss_adv = g_loss_fn(dx_)

        l2_loss_x = cycle_loss_fn(A, x__)
        l2_loss_y = cycle_loss_fn(B, y__)
        g_loss = g_loss_adv + alpha * l2_loss_x + beta * l2_loss_y
        h_loss = h_loss_adv + alpha * l2_loss_x + beta * l2_loss_y
        g_h_loss = g_loss_adv + h_loss_adv + alpha * l2_loss_x + beta * l2_loss_y
    g_h_grad = t.gradient(g_h_loss, g_net.trainable_variables + h_net.trainable_variables)
    G_optimizer.apply_gradients(zip(g_h_grad, g_net.trainable_variables + h_net.trainable_variables))
    return y_, x_, {'g_loss_adv': g_loss_adv,
                    'h_loss_adv': h_loss_adv,
                    'l2_loss_x': l2_loss_x,
                    'l2_loss_y': l2_loss_y,
                    'g_loss': g_loss,
                    'h_loss': h_loss,
                    'g_h_loss': g_h_loss}


@tf.function
def train_D(A, B, A2B, B2A):
    with tf.GradientTape() as t:
        dx = dx_net(A, training=True)
        d_fake_x = dx_net(B2A, training=True)
        # dy = dy_net(tf.concat([B,nb_classes],axis=1), training=True)
        # d_fake_y = dy_net(tf.concat([A2B,nb_classes],axis=1), training=True)
        dy = dy_net(B, training=True)
        d_fake_y = dy_net(A2B, training=True)

        A_d_loss, B2A_d_loss = d_loss_fn(dx, d_fake_x)
        dx_loss=A_d_loss+B2A_d_loss
        B_d_loss, A2B_d_loss = d_loss_fn(dy, d_fake_y)
        dy_loss=B_d_loss+A2B_d_loss
        D_A_gp = util.gradient_penalty(functools.partial(dx_net, training=True), A, B2A, mode='none')
        D_B_gp = util.gradient_penalty(functools.partial(dy_net, training=True), B, A2B, mode='none')

        D_loss = dx_loss+dy_loss + (D_A_gp + D_B_gp) * 10.0

    D_grad = t.gradient(D_loss, dx_net.trainable_variables + dy_net.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, dx_net.trainable_variables + dy_net.trainable_variables))

    return {'dx_loss': A_d_loss + B2A_d_loss,
            'dy_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp}
def train_step(A, B):
    # A2B, B2A, G_loss_dict = train_G(A, B,nb_classes)
    A2B, B2A, G_loss_dict = train_G(A, B)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A, B, A2B, B2A)

    return G_loss_dict, D_loss_dict


@tf.function
def sample(A, B):
    # A2B = g_net(tf.concat([A , nb_classes],axis=1), training=False)
    A2B = g_net(A, training=False)
    B2A = h_net(B, training=False)
    A2B2A = h_net(A2B, training=False)
    # B2A2B = g_net( tf.concat([B2A , nb_classes],axis=1), training=False)
    B2A2B = g_net(B2A, training=False)
    return A2B, B2A, A2B2A, B2A2B
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=g_net,
                                G_B2A=h_net,
                                D_A=dx_net,
                                D_B=dy_net,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           os.path.join('output','Roundtrip', 'checkpoints'),
                           max_to_keep=1)
try:  # restore checkpoint including the epoch counter
    checkpoint.restore().assert_existing_objects_matched()
    print('Checkpoint loaded successfully.')
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(os.path.join('output','Roundtrip', 'summaries', 'train'))
#
# sample
sample_dir = os.path.join('output','Roundtrip','samples_training')
# main loop

with train_summary_writer.as_default():
    for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

        # update epoch counter
        ep_cnt.assign_add(1)
        batch_idxs = len_dataset // batch_size
        # train for an epoch
        for inx in tqdm.trange(batch_idxs*10, desc='Epoch Loop'):

            x = tf.random.normal(shape=(batch_size, x_dim))
            # x_real=tf.reshape(x_real,[-1,784])
            indx = np.random.randint(low=0, high=dataset.shape[0], size=batch_size)
            data = np.reshape(dataset[indx], [batch_size, 784])
            G_loss_dict, D_loss_dict = train_step(x, data)

            # # summary
            tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
            tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
            tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

            # sample
            if G_optimizer.iterations.numpy() % 10000 == 0:
                indx = np.random.randint(low=0, high=dataset.shape[0], size=batch_size)
                data = np.reshape(dataset[indx], [batch_size, 784])
                # label = np.reshape(np.eye(10)[train_label][indx], [batch_size, 10])
                A2B, B2A, A2B2A, B2A2B = sample(x, data)
                # im=np.array([tf.reshape(A2B[0],[28,28]), tf.reshape(B2A2B[0],[28,28])])
                img_A2B = im.immerge(tf.reshape(A2B,[-1,28,28]),  n_rows=8,n_cols=8)
                img_real = im.immerge(np.reshape (data,[-1,28,28]),  n_rows=8,n_cols=8)
                img_B2A2B=im.immerge(tf.reshape(B2A2B,[-1,28,28]),  n_rows=8,n_cols=8)
                fig,((ax1,ax2),(ax3,ax4))=plt.subplots(2, 2, figsize=(4,5))
                ax1.set_title('G()')
                ax2.set_title('G(H())')
                ax3.set_title('G()')
                ax4.set_title('real')
                ax1.imshow(img_A2B, cmap='gray')
                ax2.imshow(img_B2A2B, cmap='gray')
                ax3.imshow(img_A2B, cmap='gray')
                ax4.imshow(img_real, cmap='gray')
                plt.savefig(os.path.join(sample_dir, 'iter-%09d.jpg' % G_optimizer.iterations.numpy()),dpi=600)
                plt.show()
                g_net.save('Roundtrip_model_G.h5')
                h_net.save('Roundtrip_model_H.h5')


        # save checkpoint
        checkpoint.save(ep)
    g_net.save('Roundtrip_model_G.h5')
    h_net.save('Roundtrip_model_H.h5')

