import torch
import numpy
import common.torch
import common.summary
import common.numpy
import common.quantization
from common.log import log
from imgaug import augmenters as iaa
from .training_interface import TrainingInterface


class NormalTraining(TrainingInterface):
    """
    Normal training.
    """

    def __init__(self, model, trainset, testset, optimizer, scheduler, augmentation=None, loss=common.torch.classification_loss, writer=common.summary.SummaryWriter(), cuda=False):
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
        :param augmentation: augmentation
        :type augmentation: imgaug.augmenters.Sequential
        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        :param cuda: run on CUDA device
        :type cuda: bool
        """

        assert loss is not None
        assert callable(loss)
        assert isinstance(model, torch.nn.Module)
        assert len(trainset) > 0
        assert len(testset) > 0
        assert isinstance(trainset, torch.utils.data.DataLoader)
        assert isinstance(testset, torch.utils.data.DataLoader)
        assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler)
        assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

        super(NormalTraining, self).__init__(writer)

        self.model = model
        """ (torch.nn.Module) Model. """

        parameters = list(self.model.parameters())
        self.layers = [i for i in range(len(parameters)) if parameters[i].requires_grad is True]
        """ ([int]) Layers for quantization. """

        self.trainset = trainset
        """ (torch.utils.data.DatLoader) Taining set. """

        self.testset = testset
        """ (torch.utils.data.DatLoader) Test set. """

        self.optimizer = optimizer
        """ (torch.optim.Optimizer) Optimizer. """

        self.scheduler = scheduler
        """ (torch.optim.LRScheduler) Scheduler. """

        self.augmentation = augmentation
        """ (imgaug.augmenters.Sequential) Augmentation. """

        self.cuda = cuda
        """ (bool) Run on CUDA. """

        self.loss = loss
        """ (callable) Classificaiton loss. """

        self.quantization = None
        """ (common.quantization.Quantization) Quantization. """

        self.writer.add_text('config/host', common.utils.hostname())
        self.writer.add_text('config/pid', str(common.utils.pid()))
        self.writer.add_text('config/model', self.model.__class__.__name__)
        self.writer.add_text('config/model_details', str(self.model))
        self.writer.add_text('config/trainset', self.trainset.dataset.__class__.__name__)
        self.writer.add_text('config/testset', self.testset.dataset.__class__.__name__)
        self.writer.add_text('config/optimizer', self.optimizer.__class__.__name__)
        self.writer.add_text('config/scheduler', self.scheduler.__class__.__name__)
        self.writer.add_text('config/cuda', str(self.cuda))

        self.writer.add_text('model', str(self.model))
        self.writer.add_text('optimizer', str(common.summary.to_dict(self.optimizer)))
        self.writer.add_text('scheduler', str(common.summary.to_dict(self.scheduler)))

    def quantize(self):
        """
        Quantization.
        """

        if self.quantization is not None:
            forward_model, contexts = common.quantization.quantize(self.quantization, self.model, layers=self.layers)
            return forward_model, contexts
        return self.model, None

    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

        self.model.train()
        assert self.model.training is True

        # initialize contexts
        self.quantize()

        for b, (inputs, targets) in enumerate(self.trainset):
            if self.augmentation is not None:
                if isinstance(self.augmentation, iaa.meta.Augmenter):
                    inputs = self.augmentation.augment_images(inputs.numpy())
                else:
                    inputs = self.augmentation(inputs)

            inputs = common.torch.as_variable(inputs, self.cuda)
            assert len(targets.shape) == 1
            targets = common.torch.as_variable(targets, self.cuda)
            assert len(list(targets.size())) == 1

            forward_model, contexts = self.quantize()
            assert forward_model.training is True

            self.optimizer.zero_grad()
            logits = forward_model(inputs)
            loss = self.loss(logits, targets)
            error = common.torch.classification_error(logits, targets)
            loss.backward()

            if forward_model is not self.model:
                forward_parameters = list(forward_model.parameters())
                backward_parameters = list(self.model.parameters())

                for i in range(len(forward_parameters)):
                    if backward_parameters[i].requires_grad is False:  # normalization
                        continue

                    backward_parameters[i].grad = forward_parameters[i].grad

                forward_buffers = dict(forward_model.named_buffers())
                backward_buffers = dict(self.model.named_buffers())
                for key in forward_buffers.keys():
                    if key.find('running_var') >= 0 or key.find('running_mean') >= 0 or key.find('num_batches_tracked') >= 0:
                        backward_buffers[key].data = forward_buffers[key].data

            self.optimizer.step()
            self.scheduler.step()

            global_step = epoch * len(self.trainset) + b
            self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)
            self.writer.add_scalar('train/loss', loss.item(), global_step=global_step)
            self.writer.add_scalar('train/error', error.item(), global_step=global_step)
            self.writer.add_scalar('train/confidence', torch.mean(torch.max(common.torch.softmax(logits, dim=1), dim=1)[0]).item(), global_step=global_step)

            self.progress('train %d' % epoch, b, len(self.trainset), info='loss=%g error=%g lr=%g' % (
                loss.item(),
                error.item(),
                self.scheduler.get_lr()[0],
            ))

    def test(self, epoch):
        """
        Test step.

        :param epoch: epoch
        :type epoch: int
        """

        self.optimizer.zero_grad()
        self.model.eval()
        assert self.model.training is False

        forward_model, contexts = self.quantize()
        assert forward_model.training is False

        # reason to repeat this here: use correct loss for statistics
        losses = None
        errors = None
        logits = None
        probabilities = None

        for b, (inputs, targets) in enumerate(self.testset):
            if b >= 10:
                break

            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)

            with torch.no_grad():
                outputs = forward_model(inputs)
                b_losses = self.loss(outputs, targets, reduction='none').detach().cpu().numpy()
                b_errors = common.torch.classification_error(outputs, targets, reduction='none').float().detach().cpu().numpy()

                losses = common.numpy.concatenate(losses, b_losses)
                errors = common.numpy.concatenate(errors, b_errors)
                logits = common.numpy.concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
                probabilities = common.numpy.concatenate(probabilities, common.torch.softmax(outputs, dim=1).detach().cpu().numpy())

                self.progress('test (test) %d' % epoch, b, len(self.testset), info='loss=%g error=%g' % (
                    numpy.mean(b_losses),
                    numpy.mean(b_errors)
                ))

        confidences = numpy.max(probabilities, axis=1)
        global_step = epoch  # epoch * len(self.trainset) + len(self.trainset) - 1

        self.writer.add_scalar('test/loss', numpy.mean(losses), global_step=global_step)
        self.writer.add_scalar('test/error', numpy.mean(errors), global_step=global_step)
        self.writer.add_scalar('test/logit', numpy.mean(logits), global_step=global_step)
        self.writer.add_scalar('test/confidence', numpy.mean(confidences), global_step=global_step)

        log('test (test): error=%g' % numpy.mean(errors))

        return probabilities, forward_model, contexts