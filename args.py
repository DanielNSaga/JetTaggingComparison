import torch

class Args:
    """ Inneholder alle hyperparametere for ParticleNet. """

    def __init__(self):
        # ✅ Modellparametere
        self.conv_params = [
            (16, (64, 64, 64)),
            (16, (128, 128, 128)),
            (16, (256, 256, 256))
        ]
        self.fc_params = [(0.1, 256)]
        self.input_features = 4  # pt_log, e_log, etarel, phirel
        self.output_classes = 10  # Antall klasser
        self.model_name = "ParticleNet"

        # ✅ Treningsparametere
        self.epochs = 10
        self.batch_size = 128
        self.learning_rate = 1e-3
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = "output"


