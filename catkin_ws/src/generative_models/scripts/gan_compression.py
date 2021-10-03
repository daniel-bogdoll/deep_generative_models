# [JJ]@[MW] for the GAN compress and decompress methods

from high_fidelity_generative_compression.gan_compression.src.helpers import utils
from high_fidelity_generative_compression.gan_compression.default_config import ModelModes


def compress(model, data):
    """
    Compress Input Image
    """

    # Perform entropy coding
    compressed_output = model.compress(data)
    
    return compressed_output

def decompress(model, data):
    
    # Perform entropy coding
    compressed_output = model.decompress(data)
    
    return compressed_output

def load_model(ckpt_path):
    """
    Load the GAN model
    """
    
    # Load model
    device = utils.get_device()
    logger = utils.logger_setup(logpath="/disk/no_backup/fa751/high-fidelity-generative-compression/data/originals/logs", filepath="/disk/no_backup/fa751/high-fidelity-generative-compression/data/originals/logs")
    # logger = utils.logger_setup(logpath="/disk/vanishing_data/fa401/logs/logfile", filepath="/disk/vanishing_data/fa401/logs/logfile")
    
    loaded_args, model, _ = utils.load_model(ckpt_path, logger, device, model_mode=ModelModes.EVALUATION,
        current_args_d=None, prediction=True, strict=False)
    
    # Build probability tables
    logger.info('Building hyperprior probability tables...')
    model.Hyperprior.hyperprior_entropy_model.build_tables()
    logger.info('All tables built.')


    return model