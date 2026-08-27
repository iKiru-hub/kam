import torch
import numpy as np
import argparse

import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath(__file__).split("src")[0] + "src/core")

# import utils
# import training as tr
# import visualization as vis
# from models import Autoencoder, MTL

from logger import logger


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="BTSP model")
    parser.add_argument('--num', type=int,
                        help='number of samples',
                        default=1)
    parser.add_argument('--load', action='store_true',
                        help='load session')
    parser.add_argument('--idx', type=int,
                        help='number of samples',
                        default=2)
    args = parser.parse_args()

    # """ settings """

    # if args.load:

    #     logger(f"loading idx={args.idx}...")

    #     info, autoencoder = load_session(idx=args.idx)

    #     dim_ei = info["dim_ei"]
    #     dim_ca3 = info["dim_ca3"]
    #     dim_ca1 = info["dim_ca1"]
    #     dim_eo = info["dim_eo"]

    #     num_samples = info["num_samples"]

    #     K_lat = info["K_lat"]
    #     beta = info["beta"]
    #     K = info["K"]

    #     logger("<<< Loaded session >>>")

    # # make new settings
    # if not args.load:

    #     dim_ei = 50
    #     dim_ca3 = 50 
    #     dim_ca1 = 50
    #     dim_eo = dim_ei

    #     # data settings
    #     num_samples = 300

    #     # model hyper-parameters
    #     K = 5
    #     K_lat = 18
    #     beta = 54

    #     # autoencoder
    #     autoencoder = Autoencoder(input_dim=dim_ei,
    #                               encoding_dim=dim_ca1,
    #                               K=K_lat,
    #                               beta=beta)
    #     logger(f"%Autoencoder: {autoencoder}")

    # # MTL hyper-parameters
    # K_ca3 = 22
    # alpha = 0.208

    # """ make data """

    # training_samples = utils.sparse_stimulus_generator(N=num_samples,
    #                                                    K=K,
    #                                                    size=dim_ei,
    #                                                    plot=False)
    # test_samples = utils.sparse_stimulus_generator(N=num_samples,
    #                                                K=K,
    #                                                size=dim_ei,
    #                                                plot=False)

    # # dataset for btsp
    # num_btsp_samples = args.num
    # training_sample_btsp = training_samples[np.random.choice(
    #                         range(training_samples.shape[0]),
    #                         num_btsp_samples, replace=False)]


    # logger("<<< Data generated >>>")

    # """ AUTOENCODER training """

    # # train autoencoder
    # if not args.load:
    #     epochs = 1_000
    #     loss_ae, autoencoder = tr.train_autoencoder(
    #                     training_data=training_samples,
    #                     test_data=test_samples,
    #                     model=autoencoder,
    #                     epochs=int(epochs),
    #                     batch_size=10, learning_rate=1e-3)
    #     logger(f"<<< Autoencoder trained [loss={loss_ae:.4f}] >>>")

    # # reconstruct data
    # out_ae, latent_ae = tr.reconstruct_data(
    #                                 data=training_sample_btsp,
    #                                 num=num_btsp_samples,
    #                                 model=autoencoder,
    #                                 show=False, 
    #                                 plot=False)

    # """ MTL training """

    # # get weights from the autoencoder
    # # W_ei_ca1, W_ca1_eo = autoencoder.get_weights()
    # W_ei_ca1, W_ca1_eo, B_ei_ca1, B_ca1_eo = autoencoder.get_weights(bias=True)

    # logger.debug(f"{W_ei_ca1.shape=}\n{type(W_ei_ca1)}")

    # # make model
    # model = MTL(W_ei_ca1=W_ei_ca1,
    #             W_ca1_eo=W_ca1_eo,
    #             B_ei_ca1=B_ei_ca1,
    #             B_ca1_eo=B_ca1_eo,
    #             dim_ca3=dim_ca3,
    #             K_lat=K_lat,
    #             K_out=K,
    #             K_ca3=K_ca3,
    #             beta=beta,
    #             alpha=alpha)

    # logger(f"%MTL: {model}")

    # # model with shuffled IS
    # model_rnd = MTL(W_ei_ca1=W_ei_ca1,
    #                 W_ca1_eo=W_ca1_eo,
    #                 B_ei_ca1=B_ei_ca1,
    #                 B_ca1_eo=B_ca1_eo,
    #                 dim_ca3=dim_ca3,
    #                 K_lat=K_lat,
    #                 K_out=K,
    #                 K_ca3=K_ca3,
    #                 beta=beta,
    #                 alpha=alpha,
    #                 random_IS=True)

    # logger(f"%MTL: {model}")

    # # train model | testing = training without backprop
    # epochs = 1
    # for _ in range(epochs):
    #     # _, model = utils.testing(data=training_sample_btsp,
    #     #                          model=model,
    #     #                          column=True)

    #     # _, model_rnd = utils.testing(data=training_sample_btsp,
    #     #                              model=model_rnd,
    #     #                              column=True)

    #     loss_mtl, _ = tr.testing(data=training_sample_btsp,
    #                                 model=model,
    #                                 column=True)
    #     loss_mtl_rnd, _ = tr.testing(data=training_sample_btsp,
    #                                     model=model_rnd,
    #                                     column=True)
    #     logger(f"<<< MTL trained [{loss_mtl:.3f}] >>>")
    #     logger(f"<<< MTL (s) trained [{loss_mtl_rnd:.3f}] >>>")

    # # reconstruct data
    # model.pause_lr()
    # out_mtl, latent_mtl = tr.reconstruct_data(
    #                  data=training_sample_btsp,
    #                  num=num_btsp_samples,
    #                  model=model,
    #                  column=True,
    #                  plot=False)
    # rec_loss = np.mean((training_sample_btsp - out_mtl)**2).item()

    # model_rnd.pause_lr()
    # out_mtl_rnd, latent_mtl_rnd = tr.reconstruct_data(
    #                  data=training_sample_btsp,
    #                  num=num_btsp_samples,
    #                  model=model_rnd,
    #                  column=True,
    #                  plot=False)
    # rec_loss_rnd = np.mean((training_sample_btsp - out_mtl_rnd)**2).item()


    # """ plotting """

    # # fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 5), sharex=True)
    # # is_squash = False

    # # # --- plot latent layers
    # # utils.plot_squashed_data(
    # #     data=training_sample_btsp,
    # #                          ax=ax1,
    # #                          title="Patterns", squash=is_squash)
    # # # utils.plot_squashed_data(data=latent_ae, ax=ax2,
    # # #                          title="Autoencoder",
    # # #                          squash=is_squash)
    # # utils.plot_squashed_data(data=latent_mtl_rnd, ax=ax2,
    # #                          title="shuffled $IS$",
    # #                          squash=is_squash)
    # # utils.plot_squashed_data(data=latent_mtl, ax=ax3,
    # #                          title="$IS$", squash=is_squash)

    # # fig.suptitle(f"Latent layers - $K_l=${K_lat} $\\beta=${autoencoder._beta}")

    # # --- plot data reconstruction
    # fig2, (ax12, ax22, ax32) = plt.subplots(3, 1, figsize=(15, 5), sharex=True)

    # # add more space between subplots
    # plt.subplots_adjust(hspace=0.2)

    # is_squash = False

    # vis.plot_squashed_data(data=training_sample_btsp,
    #                          ax=ax12,
    #                          title="Patterns",
    #                          squash=is_squash,
    #                          proper_title=True)
    # # utils.plot_squashed_data(data=out_ae, ax=ax22,
    # #                          title="Autoencoder",
    # #                          squash=is_squash)
    # vis.plot_squashed_data(data=out_mtl_rnd, ax=ax22,
    #                          title=f"shuffled $IS$ - reconstruction loss={rec_loss_rnd:.3f}",
    #                          squash=is_squash,
    #                          proper_title=True)
    # vis.plot_squashed_data(data=out_mtl, ax=ax32,
    #                          title=f"$IS$ - reconstruction loss={rec_loss:.3f}",
    #                          squash=is_squash,
    #                          proper_title=True)

    # # move the suptitle a bit closer to the subplots
    # fig2.suptitle(f"Reconstruction of {num_btsp_samples} stimuli",
    #               fontsize=19, y=0.93)

    # #
    # # fig3, (ax13) = plt.subplots(1, 1, figsize=(15, 5), sharex=True)
    # # cbar = plt.colorbar(
    # #     ax13.imshow(training_sample_btsp - out_mtl,
    # #                 cmap="seismic",
    # #                 aspect="auto"))
    # # ax13.set_yticks(range(num_btsp_samples))

    # # cbar.set_label("Error")
    # # ax13.set_title("pattern - mtl")
    # plt.show()



