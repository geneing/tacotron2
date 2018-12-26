import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from plotting_utils import plot_alignment_to_figure, plot_spectrogram_to_figure
from plotting_utils import plot_gate_outputs_to_figure


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, reduced_loss, grad_norm, learning_rate, duration,
                     iteration):
            self.add_scalar("training.loss", reduced_loss, iteration)
            self.add_scalar("grad.norm", grad_norm, iteration)
            self.add_scalar("learning.rate", learning_rate, iteration)
            self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_figure(
            "alignment",
            plot_alignment_to_figure(alignments[idx].data.cpu().numpy().T),
            iteration)
        self.add_figure(
            "mel_target",
            plot_spectrogram_to_figure(mel_targets[idx].data.cpu().numpy()),
            iteration)
        self.add_figure(
            "mel_predicted",
            plot_spectrogram_to_figure(mel_outputs[idx].data.cpu().numpy()),
            iteration)
        self.add_figure(
            "gate",
            plot_gate_outputs_to_figure(
                gate_targets[idx].data.cpu().numpy(),
                F.sigmoid(gate_outputs[idx]).data.cpu().numpy()),
            iteration)
