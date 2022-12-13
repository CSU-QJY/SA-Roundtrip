from __future__ import division

import functools
import os
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# import model
import numpy as np
import tf2lib as tl
import tqdm
import imlib as im
import tensorflow as tf
from tensorflow import keras
import util
import json
import Big_model


# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

def traing(args):
    print(os.getcwd())

    # ==============================================================================
    # =                                    data                                    =
    # ==============================================================================

    out_dim = np.sqrt(args.y[0] * args.y[1]).astype('int32')
    # A_img_paths = glob.glob(os.path.join('E:\\Tensorflow1.0\\test_3\\TF2_Roundtrip\\data', '*.png'))
    #
    # dataset, shape, len_dataset = util.make_zip_dataset(A_img_paths,args.dataset, args.batch_size)
    dataset, img_shape, len_dataset = util.load_mnist(args.dataset, args.batch_size)
    # ==============================================================================
    # =                                   models                                   =
    # ==============================================================================

    g_net = Big_model.Generator_img(input_dim=args.x, name='g_net', nb_layers=3, nb_units=args.dim * 4)
    h_net = Big_model.Encoder_img(input_dim=args.y, output_dim=args.x, name='h_net', nb_layers=2,
                                  nb_units=args.dim)
    dx_net = Big_model.Discriminator(input_dim=args.x, name='dx_net', nb_layers=2, nb_units=args.dim*4)
    dy_net = Big_model.Discriminator_img(input_dim=args.y, name='dy_net', nb_layers=2, nb_units=args.dim)

    d_loss_fn, g_loss_fn = util.get_lsgan_losses_fn()
    cycle_loss_fn = tf.losses.MeanSquaredError()
    A2B_pool = util.ItemPool(50)
    B2A_pool = util.ItemPool(50)

    # G_lr_scheduler = tf.keras.optimizers.schedules.serialize(args.lr, args.epoch_decay, 0.96)
    # D_lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(args.lr, args.epoch_decay, 0.96)
    G_optimizer = keras.optimizers.Nadam(learning_rate=args.lr, beta_1=args.adam_beta_1, beta_2=args.adam_beta_2)
    D_optimizer = keras.optimizers.Nadam(learning_rate=args.lr, beta_1=args.adam_beta_1, beta_2=args.adam_beta_2)

    # ==============================================================================
    # =                                 train function                             =
    # ==============================================================================
    @tf.function
    def train_G(noise, image):
        with tf.GradientTape() as t:
            fake_img = g_net(noise, training=True)
            latent_code_real = h_net(image, training=True)
            e_g_img = g_net(latent_code_real, training=True)
            g_e_noise = h_net(fake_img, training=True)
            fake_f_score = dy_net(fake_img, training=True)
            fake_h_score = dx_net(latent_code_real, training=True)
            l2_loss_x = cycle_loss_fn(noise, g_e_noise)
            l2_loss_y = cycle_loss_fn(image, e_g_img)
            g_loss = g_loss_fn(fake_f_score)
            e_loss = g_loss_fn(fake_h_score)
            g_e_loss = g_loss + e_loss + args.alpha * l2_loss_x + args.beta * l2_loss_y
        grad_gen_en = t.gradient(g_e_loss, g_net.trainable_variables + h_net.trainable_variables)
        G_optimizer.apply_gradients(zip(grad_gen_en, g_net.trainable_variables + h_net.trainable_variables))
        return fake_img, latent_code_real, \
               {'G_E_loss': g_e_loss, 'g_loss': g_loss, 'e_loss': e_loss, 'lg_loss': l2_loss_y, 'lh_loss': l2_loss_x}

    @tf.function
    def train_D(noise, image, fake_img, latent_code_real):
        with tf.GradientTape() as t:
            real_f_score = dy_net(image, training=True)
            fake_f_score = dy_net(fake_img, training=True)
            fake_h_score = dx_net(latent_code_real, training=True)
            real_h_score = dx_net(noise, training=True)
            A_d_loss, B2A_d_loss = d_loss_fn(real_f_score, fake_f_score)
            B_d_loss, A2B_d_loss = d_loss_fn(real_h_score, fake_h_score)
            f_loss = A_d_loss + B2A_d_loss
            h_loss = B_d_loss + A2B_d_loss
            # gp = util.gradient_penalty(functools.partial(dy_net, training=True), image, fake_img, 'dragan') + util.gradient_penalty(functools.partial(dx_net, training=True), noise, latent_code_real, 'dragan')

            d_loss = f_loss + h_loss
        grad_disc = t.gradient(d_loss, dy_net.trainable_variables + dx_net.trainable_variables)
        D_optimizer.apply_gradients(zip(grad_disc, dy_net.trainable_variables + dx_net.trainable_variables))

        return {'D_loss': d_loss, 'f_loss': f_loss, 'h_loss': h_loss}

    def train_step(image):
        for i in range(args.n_d):
            noise = tf.random.normal(shape=((args.batch_size,).__add__(args.x)))
            fake_img, latent_code_real, G_loss_dict = train_G(noise, image)
        fake_img, latent_code_real = A2B_pool(fake_img), B2A_pool(latent_code_real)
        D_loss_dict = train_D(noise, image, fake_img, latent_code_real)
        Loss_dict = dict(G_loss_dict, **D_loss_dict)
        return Loss_dict

    @tf.function
    def sample(A, B):
        A2B = g_net(A, training=False)
        B2A = h_net(B, training=False)
        A2B2A = h_net(A2B, training=False)
        B2A2B = g_net(B2A, training=False)
        return A2B, B2A, A2B2A, B2A2B

    # ==============================================================================
    # =                         location and checkpoint                            =
    # ==============================================================================
    dir = os.path.join('output', f'{args.x[-1]}_{args.dataset}_{args.model}')

    if not os.path.exists(dir):
        os.makedirs(dir)

    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

    # checkpoint
    checkpoint = tl.Checkpoint(dict(G_A2B=g_net,
                                    G_B2A=h_net,
                                    D_A=dx_net,
                                    D_B=dy_net,
                                    G_optimizer=G_optimizer,
                                    D_optimizer=D_optimizer,
                                    ep_cnt=ep_cnt),
                               dir,
                               max_to_keep=3)
    try:  # restore checkpoint including the epoch counter
        checkpoint.restore().assert_existing_objects_matched()
        print('Checkpoint loaded successfully.')
    except Exception as e:
        print(e)

    # summary
    train_summary_writer = tf.summary.create_file_writer(dir)
    json_hp = json.dumps(args.__dict__, indent=2)
    json_hp = "".join("\t" + line for line in json_hp.splitlines(True))

    # main loop
    # ==============================================================================
    # =                                 train step                                 =
    # ==============================================================================

    with train_summary_writer.as_default():
        tf.summary.text("run_params", json_hp, step=0)
        # tf.summary.trace_on(graph=True, profiler=True)

        for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):
            if ep < ep_cnt:
                continue

            # update epoch counter
            ep_cnt.assign_add(1)
            # batch_idxs = len_dataset // batch_size
            os.system('nvidia-smi')
            # for inx in tqdm.trange(batch_idxs * 5, desc='Epoch Loop'):
            # args.alpha = 10 * 0.99 ** ep
            # args.beta = 10 * 0.99 ** ep
            print(args.beta)
            with tqdm.tqdm(dataset, total=len_dataset) as t:
                for data in dataset:
                    # x = tf.random.normal(shape=(batch_size,x_dim))
                    # x = tf.random.truncated_normal(shape=(args.batch_size, args.x))
                    # data = tf.reshape(data, [-1, args.y])
                    loss_dict = train_step(data)
                    t.set_description(f'Epoch {ep}')

                    t.set_postfix(ordered_dict={'G_E_loss': loss_dict['G_E_loss'].numpy(),
                                                'D_loss': loss_dict['D_loss'].numpy()})
                    # # summary
                    tl.summary(loss_dict, step=G_optimizer.iterations, scope='losses')

                    t.update(1)
                    train_summary_writer.flush()

                    # sample
                    # if G_optimizer.iterations.numpy() % (args.epoch_decay * 2) == 0:
                    if G_optimizer.iterations.numpy() % len_dataset == 0:
                        z = tf.random.normal(shape=((args.batch_size,).__add__(args.x)))
                        A2B, B2A, A2B2A, B2A2B = sample(z, data)

                        distribution = {'Real': z, 'H': B2A, 'H_G': A2B2A}

                        img_A2B = {'G': im.immerge(np.reshape(A2B[:16], [-1, out_dim, out_dim, 1]), n_rows=4, n_cols=4)}
                        img_real = {
                            'Real': im.immerge(np.reshape(data[:16], [-1, out_dim, out_dim, 1]), n_rows=4, n_cols=4)}
                        img_B2A2B = {
                            'G(H)': im.immerge(np.reshape(B2A2B[:16], [-1, out_dim, out_dim, 1]), n_rows=4, n_cols=4)}
                        img = dict(img_A2B, **img_real, **img_B2A2B)

                        tl.summary(distribution, step=G_optimizer.iterations, scope='Distribution')
                        tl.summary(img, step=G_optimizer.iterations, scope='Images')

                        g_net.save(f'{dir}/G_{args.x[-1]}_{args.dataset}_{args.model}_{G_optimizer.iterations.numpy()}.h5')
                        h_net.save(f'{dir}/H_{args.x[-1]}_{args.dataset}_{args.model}_{G_optimizer.iterations.numpy()}.h5')
                        checkpoint.save(ep)

            # save checkpoint

    # tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir=dir)
    g_net.save(f'G_{args.x[-1]}_{args.dataset}_{args.model}.h5')
    h_net.save(f'H_{args.x[-1]}_{args.dataset}_{args.model}.h5')
