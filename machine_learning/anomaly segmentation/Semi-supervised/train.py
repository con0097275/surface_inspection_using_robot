

import os
import argparse
import tensorflow as tf
from tensorflow import keras
from autoencoder.autoencoder import AutoEncoder
from processing.preprocessing import Preprocessor
from processing.utils import printProgressBar as printProgressBar
from processing import utils
from processing import postprocessing
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
Valid combinations for input arguments for architecture, color_mode and loss:
                        +----------------+----------------+
                        |       Model Architecture        |
                        +----------------+----------------+
                        |  mvtecCAE      |   ResnetCAE    |
                        |  baselineCAE   |                |
                        |  inceptionCAE  |                |
========================+================+================+
        ||              |                |                |
        ||   grayscale  |    ssim, l2    |    ssim, l2    |
Color   ||              |                |                |
Mode    ----------------+----------------+----------------+
        ||              |                |                |
        ||      RGB     |    mssim, l2   |    mssim, l2   |
        ||              |                |                |
--------+---------------+----------------+----------------+
"""


def check_arguments(architecture, color_mode, loss):
    if loss == "mssim" and color_mode == "grayscale":
        raise ValueError("MSSIM works only with rgb images")
    if loss == "ssim" and color_mode == "rgb":
        raise ValueError("SSIM works only with grayscale images")
    return


def main(args):

    # get parsed arguments from user
    # input_dir = args.input_dir
    # architecture = args.architecture
    # color_mode = args.color
    # loss = args.loss
    # batch_size = args.batch
    
    input_dir = "crackdataset/wallsample"
    architecture = "inceptionCAE"
    color_mode = "grayscale"
    loss = "ssim"
    batch_size = 32
    
    # check arguments
    check_arguments(architecture, color_mode, loss)

    # get autoencoder
    autoencoder = AutoEncoder(input_dir, architecture, color_mode, loss, batch_size)

    # load data as generators that yield batches of preprocessed images
    preprocessor = Preprocessor(
        input_directory=input_dir,
        rescale=autoencoder.rescale,
        shape=autoencoder.shape,
        color_mode=autoencoder.color_mode,
        preprocessing_function=autoencoder.preprocessing_function,
    )
    train_generator = preprocessor.get_train_generator(
        batch_size=autoencoder.batch_size, shuffle=True
    )
    validation_generator = preprocessor.get_val_generator(
        batch_size=autoencoder.batch_size, shuffle=True
    )

    # find best learning rates for training
    autoencoder.find_lr_opt(train_generator, validation_generator)
    
    # ##################### test find_lr_opt:
    # # initialize learner object
    # import ktrain
    # autoencoder.learner = ktrain.get_learner(
    #     model=autoencoder.model,
    #     train_data=train_generator,
    #     val_data=validation_generator,
    #     batch_size=autoencoder.batch_size,
    # )

    # # simulate training while recording learning rate and loss
    # logger.info("initiating learning rate finder to determine best learning rate.")

    # autoencoder.learner.lr_find(
    #     start_lr=autoencoder.start_lr,
    #     lr_mult=1.01,
    #     max_epochs=autoencoder.lr_max_epochs,
    #     # max_epochs=14,      #edit
    #     stop_factor=6,
    #     verbose=autoencoder.verbose,
    #     show_plot=True,
    #     restore_weights_only=True,
    # )
    # ############## autoencoder.ktrain_lr_estimate()
    # import numpy as np
    # losses = np.array(autoencoder.learner.lr_finder.losses)
    # lrs = np.array(autoencoder.learner.lr_finder.lrs)
    # # minimum loss devided by 10
    # autoencoder.ml_i = autoencoder.learner.lr_finder.ml
    # autoencoder.lr_ml_10 = lrs[autoencoder.ml_i] / 10
    # logger.info(f"lr with minimum loss divided by 10: {autoencoder.lr_ml_10:.2E}")
    # try:
    #     min_loss_i = np.argmin(losses)
    #     autoencoder.lr_ml_10_i = np.argwhere(lrs[:min_loss_i] > autoencoder.lr_ml_10)[0][0]
    # except:
    #     # print("ccc")
    #     autoencoder.lr_ml_10_i = None
    # # minimum gradient
    # autoencoder.lr_mg_i = autoencoder.learner.lr_finder.mg
    # if autoencoder.lr_mg_i is not None:
    #     autoencoder.lr_mg = lrs[autoencoder.lr_mg_i]
    #     logger.info(f"lr with minimum numerical gradient: {self.lr_mg:.2E}")
        
    # ##################  autoencoder.custom_lr_estimate()
    # losses = np.array(autoencoder.learner.lr_finder.losses)
    # lrs = np.array(autoencoder.learner.lr_finder.lrs)
    # # find optimal learning rate
    # min_loss = np.amin(losses)
    # min_loss_i = np.argmin(losses)
    # # retrieve segment containing decreasing losses
    # segment = losses[: min_loss_i + 1]
    # max_loss = np.amax(segment)
    # # compute optimal loss
    # optimal_loss = max_loss - autoencoder.lrf_decrease_factor * (max_loss - min_loss)
    # # get optimal learning rate index (corresponding to optimal loss) and value
    # autoencoder.lr_opt_i = np.argwhere(segment <= optimal_loss)[0][0]
    # autoencoder.lr_opt = float(lrs[autoencoder.lr_opt_i])
    # # get base learning rate
    # autoencoder.lr_base_i = np.argwhere(lrs[:min_loss_i+1] >= autoencoder.lr_opt / 10)[0][0]
    # autoencoder.lr_base = float(lrs[autoencoder.lr_base_i])
    # # log to console
    # logger.info(f"custom base learning rate: {lrs[autoencoder.lr_base_i]:.2E}")
    # logger.info(f"custom optimal learning rate: {lrs[autoencoder.lr_opt_i]:.2E}")
    # logger.info("learning rate finder complete.")
    
    
    
    # autoencoder.lr_find_plot(n_skip_beginning=10, n_skip_end=1, save=True)
        
    ####################
    
    # autoencoder.lr_base_i=3
    # autoencoder.lr_base = float(lrs[autoencoder.lr_base_i])
    # autoencoder.lr_opt_i=4
    # autoencoder.lr_opt = float(lrs[autoencoder.lr_opt_i])
    # # train
    # autoencoder.fit(lr_opt=autoencoder.lr_opt)

    # ################ test fit:
    # tensorboard_cb = keras.callbacks.TensorBoard(
    #     log_dir=autoencoder.log_dir, write_graph=True, update_freq="epoch"
    # )
    # # Print command to paste in browser for visualizing in Tensorboard
    # logger.info(
    #     "run the following command in a seperate terminal to monitor training on tensorboard:"
    #     + "\ntensorboard --logdir={}\n".format(autoencoder.log_dir)
    # )
    # assert autoencoder.learner.model is autoencoder.model

    # # fit model using Cyclical Learning Rates
    # autoencoder.hist = autoencoder.learner.autofit(
    #     lr=autoencoder.lr_opt,
    #     epochs=14,   #edit: default None
    #     early_stopping=autoencoder.early_stopping,
    #     reduce_on_plateau=autoencoder.reduce_on_plateau,
    #     reduce_factor=2,
    #     cycle_momentum=True,
    #     max_momentum=0.95,
    #     min_momentum=0.85,
    #     monitor="val_loss",
    #     checkpoint_folder=None,
    #     verbose=autoencoder.verbose,
    #     callbacks=[tensorboard_cb],
    # )
    
    
    
    # #############################

    # save model
    autoencoder.fit(lr_opt=autoencoder.lr_opt)
    autoencoder.save()

    if args.inspect:
        # -------------- INSPECTING VALIDATION IMAGES --------------
        logger.info("generating inspection plots of validation images...")

        # create a directory to save inspection plots
        inspection_val_dir = os.path.join(autoencoder.save_dir, "inspection_val")
        if not os.path.isdir(inspection_val_dir):
            os.makedirs(inspection_val_dir)

        inspection_val_generator = preprocessor.get_val_generator(
            batch_size=autoencoder.learner.val_data.samples, shuffle=False
        )

        imgs_val_input = inspection_val_generator.next()[0]
        filenames_val = inspection_val_generator.filenames

        # get reconstructed images (i.e predictions) on validation dataset
        logger.info("reconstructing validation images...")
        imgs_val_pred = autoencoder.model.predict(imgs_val_input)

        # instantiate TensorImages object to compute validation resmaps
        tensor_val = postprocessing.TensorImages(
            imgs_input=imgs_val_input,
            imgs_pred=imgs_val_pred,
            vmin=autoencoder.vmin,
            vmax=autoencoder.vmax,
            method=autoencoder.loss,
            dtype="float64",
            filenames=filenames_val,
        )

        # generate and save inspection validation plots
        tensor_val.generate_inspection_plots(
            group="validation", save_dir=inspection_val_dir
        )

        # -------------- INSPECTING TEST IMAGES --------------
        logger.info("generating inspection plots of test images...")

        # create a directory to save inspection plots
        inspection_test_dir = os.path.join(autoencoder.save_dir, "inspection_test")
        if not os.path.isdir(inspection_test_dir):
            os.makedirs(inspection_test_dir)

        nb_test_images = preprocessor.get_total_number_test_images()

        inspection_test_generator = preprocessor.get_test_generator(
            batch_size=nb_test_images, shuffle=False
        )

        imgs_test_input = inspection_test_generator.next()[0]
        filenames_test = inspection_test_generator.filenames

        # get reconstructed images (i.e predictions) on validation dataset
        logger.info("reconstructing test images...")
        imgs_test_pred = autoencoder.model.predict(imgs_test_input)

        # instantiate TensorImages object to compute test resmaps
        tensor_test = postprocessing.TensorImages(
            imgs_input=imgs_test_input,
            imgs_pred=imgs_test_pred,
            vmin=autoencoder.vmin,
            vmax=autoencoder.vmax,
            method=autoencoder.loss,
            dtype="float64",
            filenames=filenames_test,
        )

        # generate and save inspection test plots
        tensor_test.generate_inspection_plots(
            group="test", save_dir=inspection_test_dir
        )

    logger.info("done.")
    return


if __name__ == "__main__":
    # create parser
    parser = argparse.ArgumentParser(
        description="Train an AutoEncoder on an image dataset.",
        epilog="Example usage: python3 train.py -d mvtec/capsule -a mvtec2 -b 8 -l ssim -c grayscale",
    )
    parser.add_argument(
        "-d",
        "--input-dir",
        type=str,
        required=True,
        metavar="",
        help="directory containing training images",
    )

    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        required=False,
        metavar="",
        choices=["mvtecCAE", "baselineCAE", "inceptionCAE", "resnetCAE", "skipCAE"],
        default="mvtec2",
        help="architecture of the model to use for training: 'mvtecCAE', 'baselineCAE', 'inceptionCAE', 'resnetCAE' or 'skipCAE'",
    )

    parser.add_argument(
        "-c",
        "--color",
        type=str,
        required=False,
        metavar="",
        choices=["rgb", "grayscale"],
        default="grayscale",
        help="color mode for preprocessing images before training: 'rgb' or 'grayscale'",
    )

    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        required=False,
        metavar="",
        choices=["mssim", "ssim", "l2"],
        default="ssim",
        help="loss function to use for training: 'mssim', 'ssim' or 'l2'",
    )

    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        required=False,
        metavar="",
        default=8,
        help="batch size to use for training",
    )

    parser.add_argument(
        "-i",
        "--inspect",
        action="store_true",
        help="generate inspection plots after training",
    )

    args = parser.parse_args()
    if tf.test.is_gpu_available():
        logger.info("GPU was detected...")
    else:
        logger.info("No GPU was detected. CNNs can be very slow without a GPU...")
    logger.info("Tensorflow version: {} ...".format(tf.__version__))
    logger.info("Keras version: {} ...".format(keras.__version__))
    main(args)

# Examples of commands to initiate training with mvtec architecture

# python3 train.py -d mvtec/capsule -a mvtecCAE -b 8 -l ssim -c grayscale --inspect
# python3 train.py -d mvtec/hazelnut -a resnetCAE -b 8 -l mssim -c rgb --inspect
# python3 train.py -d mvtec/pill -a inceptionCAE -b 8 -l mssim -c rgb --inspect
