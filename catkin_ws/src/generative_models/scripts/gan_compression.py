# [JJ]@[MW] for the GAN compress and decompress methods

from high_fidelity_generative_compression.gan_compression.src.helpers import utils
from high_fidelity_generative_compression.gan_compression.default_config import ModelModes


def compress(model, data):
    """
    Compress input image to latent representation
    """

    # Perform entropy coding
    compressed_output = model.compress(data)
    

    return compressed_output

def decompress(model, data):
    """
    Decompress latent representation to reconstructed image
    """
    # Perform entropy coding
    compressed_output = model.decompress(data)
    

    return compressed_output

def load_model(ckpt_path):
    """
    Load the GAN model
    """
    
    # Get device
    device = utils.get_device()
    
    # Initialize logger
    logger = utils.logger_setup(logpath="PATH_TO_LOG", filepath="PATH_TO_LOG")
    
    # Load model
    loaded_args, model, _ = utils.load_model(ckpt_path, logger, device, model_mode=ModelModes.EVALUATION,
        current_args_d=None, prediction=True, strict=False)
    
    # Build probability tables
    logger.info('Building hyperprior probability tables...')
    model.Hyperprior.hyperprior_entropy_model.build_tables()
    logger.info('All tables built.')


    return model