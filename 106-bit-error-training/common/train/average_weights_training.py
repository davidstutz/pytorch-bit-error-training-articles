import torch
import common.torch
import common.summary
import common.numpy
import common.progress
import common.calibration
from imgaug import augmenters as iaa
from .adversarial_weights_training import AdversarialWeightsTraining


class AverageWeightsTraining(AdversarialWeightsTraining):
    """
    Adversarial training.
    """

    def __init__(self, model, trainset, testset, optimizer, scheduler, attack, objective, operators=None, augmentation=None, loss=common.torch.classification_loss, writer=common.summary.SummaryWriter(), cuda=False):
        """
        Constructor.

        :param model: model
        :type model: torch.nn.Module
        :param trainset: training set
        :type trainset: torch.utils.data.DataLoader
        :param testset: test set
        :type testset: torch.utils.data.DataLoader
        :param optimizer: optimizer
        :type optimizer: torch.optim.Optimizer
        :param scheduler: scheduler
        :type scheduler: torch.optim.LRScheduler
        :param attack: attack
        :type attack: attacks.Attack
        :param objective: objective
        :type objective: attacks.Objective
        :param augmentation: augmentation
        :type augmentation: imgaug.augmenters.Sequential
        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        :param cuda: run on CUDA device
        :type cuda: bool
        """

        super(AverageWeightsTraining, self).__init__(model, trainset, testset, optimizer, scheduler, attack, objective, operators, augmentation, loss, writer, cuda)

    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

        assert not (self.average_statistics and self.adversarial_statistics)

        if self.curriculum is None:
            self.population = 1

        # initialize contexts
        self.quantize()

        for b, (inputs, targets) in enumerate(self.trainset):
            if self.augmentation is not None:
                if isinstance(self.augmentation, iaa.meta.Augmenter):
                    inputs = self.augmentation.augment_images(inputs.numpy())
                else:
                    inputs = self.augmentation(inputs)

            # before permutation!
            # works with enumerate() similar to data loader.
            batchset = [(inputs, targets)]

            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)

            self.project()
            forward_model, contexts = self.quantize()

            self.model.train()
            forward_model.train()
            self.optimizer.zero_grad()
            logits = forward_model(inputs)

            loss = self.loss(logits, targets)
            error = common.torch.classification_error(logits, targets)

            loss.backward()

            if forward_model is not self.model:
                forward_parameters = list(forward_model.parameters())
                backward_parameters = list(self.model.parameters())
                for j in range(len(forward_parameters)):
                    if backward_parameters[j].requires_grad is False:  # normalization
                        continue

                    # no addition! for quantization parameter.grad is non
                    # without quantization this is essentially an expensive no-op
                    #print('clean', torch.max(forward_parameters[j].grad.data).item(), torch.min(forward_parameters[j].grad.data).item())
                    gradient = forward_parameters[j].grad.data
                    #gradient = torch.div(gradient, torch.norm(gradient.view(-1), p=2))
                    backward_parameters[j].grad = gradient

                if not self.adversarial_statistics:
                    forward_buffers = dict(forward_model.named_buffers())
                    backward_buffers = dict(self.model.named_buffers())
                    for key in forward_buffers.keys():
                        if key.find('running_var') >= 0 or key.find('running_mean') >= 0 or key.find('num_batches_tracked') >= 0:
                           backward_buffers[key].data = forward_buffers[key].data
                        #if key.find('running_var') >= 0 or key.find('running_mean') >= 0:
                        #    print('clean', key, torch.mean(backward_buffers[key].data.float()).item())

            self.model.eval()
            forward_model.eval()
            global_step = epoch * len(self.trainset) + b

            population_norm = 0
            population_perturbed_loss = 0
            population_perturbed_error = 0

            if self.reset_iterations % (b + 1) == 0:
                self.objective.reset()

            for i in range(self.population):
                # optional: set objective targets or so?
                #self.attack.progress = common.progress.ProgressBar()
                perturbed_model = self.attack.run(forward_model, batchset, self.objective)

                # This is a perturbation based on the original eval model!
                perturbed_model.train()
                common.calibration.reset(perturbed_model)
                common.calibration.momentum(perturbed_model, 1)
                perturbed_logits = perturbed_model(inputs, operators=self.operators)

                perturbed_loss = self.loss(perturbed_logits, targets)
                perturbed_error = common.torch.classification_error(perturbed_logits, targets)
                #print(loss.item(), error.item())

                population_perturbed_loss += perturbed_loss.item()
                population_perturbed_error += perturbed_error.item()

                #perturbed_loss = torch.min(perturbed_loss, torch.ones_like(perturbed_loss)*3)
                perturbed_loss.backward()

                # take average of gradients
                backward_parameters = list(self.model.parameters())
                perturbed_parameters = list(perturbed_model.parameters())
                for j in range(len(backward_parameters)):
                    if backward_parameters[j].requires_grad is False:  # normalization
                        continue

                    # should already be clean parameter grads here!
                    gradient = perturbed_parameters[j].grad.data
                    backward_parameters[j].grad.data += torch.clamp(gradient, min=-self.gradient_clipping, max=self.gradient_clipping)

                if self.average_statistics or self.adversarial_statistics:
                    perturbed_buffers = dict(perturbed_model.named_buffers())
                    backward_buffers = dict(self.model.named_buffers())
                    for key in perturbed_buffers.keys():
                        if key.find('running_var') >= 0 or key.find('running_mean') >= 0:
                            #print('perturbed', key, torch.mean(backward_buffers[key].data.float()).item(), torch.mean(perturbed_buffers[key].data.float()).item())
                            backward_buffers[key].data = 0.1*perturbed_buffers[key].data + 0.9*backward_buffers[key].data
                        if key.find('num_batches_tracked') >= 0:
                            backward_buffers[key].data += 1

                self.writer.add_scalar('train/adversarial_loss%d' % i, perturbed_loss.item(), global_step=global_step)
                self.writer.add_scalar('train/adversarial_error%d' % i, perturbed_error.item(), global_step=global_step)

                if self.attack.norm is not None:
                    norm = self.attack.norm(forward_model, perturbed_model, self.attack.layers, self.quantization, contexts)
                    population_norm += norm
                    self.writer.add_scalar('train/adversarial_norm%d' % i, norm, global_step=global_step)
                    for j in range(len(self.attack.norm.norms)):
                        self.writer.add_scalar('train/adversarial_norms%d/%d' % (i, j), self.attack.norm.norms[j], global_step=global_step)

            if self.population > 0:
                population_norm /= self.population
                population_perturbed_loss /= self.population
                population_perturbed_error /= self.population
                for parameter in self.model.parameters():
                    if parameter.requires_grad is False:  # normalization
                        continue
                    parameter.grad.data /= (self.population + 1)

            self.model.train()
            forward_model.train()

            self.optimizer.step()
            self.scheduler.step()

            curriculum_logs = dict()
            if self.curriculum is not None:
                self.population, curriculum_logs = self.curriculum(self.attack, loss, population_perturbed_loss, epoch)
                for curriculum_key, curriculum_value in curriculum_logs.items():
                    self.writer.add_scalar('train/curriculum/%s' % curriculum_key, curriculum_value, global_step=global_step)

            self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)

            self.writer.add_scalar('train/loss', loss.item(), global_step=global_step)
            self.writer.add_scalar('train/error', error.item(), global_step=global_step)
            self.writer.add_scalar('train/confidence', torch.mean(torch.max(common.torch.softmax(logits, dim=1), dim=1)[0]).item(), global_step=global_step)

            self.progress('train %d' % epoch, b, len(self.trainset), info='loss=%g err=%g advloss=%g adverr=%g advnorm=%g lr=%g pop=%d curr=%s' % (
                loss.item(),
                error.item(),
                population_perturbed_loss,
                population_perturbed_error,
                population_norm,
                self.scheduler.get_lr()[0],
                self.population,
                str(list(curriculum_logs.values())),
            ))
