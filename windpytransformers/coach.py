# -*- coding: UTF-8 -*-
""""
Created on 26.02.20
Module with class for model training.

:author:     Martin Dočekal
"""

import time
from abc import ABC, abstractmethod
from enum import Enum
from math import ceil

import torch
from torch.utils.data import Sampler, DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedModel
from typing import Optional, Tuple, Callable, Dict, Generator, Any, TypeVar, Generic

from windpytorchutils.optimizers import OptimizerCreator, BERTAdamWOptimizerCreator
from windpytorchutils.samplers import IndicesSubsampler
from windpyutils.logger import Logger


class CoachTransformersConfig(object):
    """
    Config structure for CoachTransformers models.
    """

    def __init__(self, mixed_precision=False):
        self.mixed_precision = mixed_precision


class CoachTransformersValidateCallback(ABC):
    """
    Functor for model validation.
    """

    @abstractmethod
    def __call__(self, epoch: int, averageLoss: float, save: bool = True) -> bool:
        """
        Start the validation.

        :param epoch: Number of actual epoch.
        :type epoch: int
        :param averageLoss: Average loss in the epoch.
        :type averageLoss: float
        :param save: True enables model saving.
        :type save: bool
        :return: True training should stop. False training should continue.
        :rtype: bool
        """
        pass


ConfigT = TypeVar('ConfigT', bound='CoachTransformersConfig')
"""
Generic type for configuration.
"""


class CoachTransformers(Generic[ConfigT]):
    """
    This class gathers boilerplate code that is usually used for model training of transformers models from huggingface implementations https://huggingface.co/transformers/#.
    """

    BAR_INTERVAL = 10
    """Minimal number of seconds for update."""

    class Fp16OptLvls(Enum):
        """
        Optimization level for 16 bit float mixed precession.

        The docs strings are from (inspired from) the: https://nvidia.github.io/apex/amp.html
        """

        O0 = "O0"
        """
        Title: FP32 training
        Description:
            Because the model is probably in FP32 this means no operation
        """

        O1 = "O1"
        """
        Title: Mixed Precision (recommended for typical use [and default for this KeywordsExtractor])
        Description: 
            Patch all Torch functions and Tensor methods to cast their inputs according to a whitelist-blacklist model. 
            Whitelist ops (for example, Tensor Core-friendly ops like GEMMs and convolutions) are performed in FP16. 
            Blacklist ops that benefit from FP32 precision (for example, softmax) are performed in FP32. 
            O1 also uses dynamic loss scaling, unless overridden.
        """

        O2 = "O2"
        """
        Title: “Almost FP16” Mixed Precision
        Description:
            O2 casts the model weights to FP16, patches the model’s forward method 
                to cast input data to FP16, 
                keeps batchnorms in FP32, 
                maintains FP32 master weights, 
                updates the optimizer’s param_groups so that the optimizer.step() acts directly on the FP32 weights 
                (followed by FP32 master weight->FP16 model weight copies if necessary), 
                and implements dynamic loss scaling (unless overridden). 
            Unlike O1, O2 does not patch Torch functions or Tensor methods.
        """

        O3 = "O3"
        """
        Title: FP16 training
        Description: 
            O3 may not achieve the stability of the true mixed precision options O1 and O2. 
            However, it can be useful to establish a speed baseline for your model, 
            against which the performance of O1 and O2 can be compared. 
            If your model uses batch normalization, to establish “speed of light” you can try O3 with the additional 
            property override keep_batchnorm_fp32=True (which enables cudnn batchnorm, as stated earlier).
        """

    def __init__(self, model: PreTrainedModel, trainInput2DataMap: Dict[str, int], useInput2DataMap: Dict[str, int],
                 mixedPrecision: bool = False):
        """
        Initialization of new/pretrained entity linker.

        :param model: Model you want to work with.
            !!! WARNING make sure that the model:
                    is using compatible configuration with CoachTransformersConfig
                    returns it's loss as first value when trained
        :type model: PreTrainedModel
        :param trainInput2DataMap: This dictionary contains mapping of model inputs to dataset offsets of appropriate data types
            in train phase.
        :type trainInput2DataMap: Dict[str, int]
        :param useInput2DataMap: This dictionary contains mapping of model inputs to dataset offsets of appropriate data types
            in use phase.
        :type useInput2DataMap: Dict[str, int]
        :param mixedPrecision: Activates mixed precision.
        :type mixedPrecision: bool
        :raises:
            AttributeError: On mixedPrecision mismatch.
            AssertionError: When invalid config is used.
        """

        assert isinstance(model.config, CoachTransformersConfig)

        self.model = model

        self.trainInput2DataMap = trainInput2DataMap
        self.useInput2DataMap = useInput2DataMap
        # Set the mixed precision optimization level
        # More info can be found at: https://nvidia.github.io/apex/amp.html

        self.FP16_MIX_PREC_OPT_LVL = self.Fp16OptLvls.O1
        if self._mixedPrecisionActivated and not mixedPrecision:
            AttributeError("The model is already in mixed precision. You can not switch it to full precision.")
        self._shouldActivateMixedPrecision = mixedPrecision

    @property
    def _mixedPrecisionActivated(self):
        """
        Flag that determines if model is in mixed precision.

        :return: True if mixed precision is activate. False otherwise.
        :rtype: bool
        """
        return self.config.mixed_precision

    @_mixedPrecisionActivated.setter
    def _mixedPrecisionActivated(self, isMixed: bool):
        """
        Sets the mixed precission flag.

        :param isMixed: True mixed precision is activated.
        :type isMixed: bool
        """

        self.config.mixed_precision = isMixed

    @property
    def config(self) -> ConfigT:
        """
        Config of the model.

        :return: The config of the model.
        :rtype: ConfigT
        """
        return self.model.config

    def save(self, p: str):
        """
        Saves meter to given folder.

        :param p: Path to folder where the meter should be saved.
        :type p: str
        """

        self.model.save_pretrained(p)

    def _initModelAndLoader(self, dataset: torch.utils.data.Dataset, batchSize: int, shuffle: bool = True,
                            optimizerCreator: Optional[OptimizerCreator] = None, loadingWorkers: int = 1,
                            forceDevice: str = None, dataLoaderCollate: Optional[Callable] = None,
                            modelTrain: bool = False, sampler: Optional[Sampler] = None) \
            -> Tuple[DataLoader, torch.optim.Optimizer, torch.device]:
        """
        Initialization method that gathers common operations that are done at start of train, eval or inference phase.

        :param dataset: Dataset that should be used.
        :type dataset: torch.utils.data.Dataset
        :param batchSize: Size of one batch.
        :type batchSize: int
        :param shuffle: True reshuffles data on every new iteration trough dataset.
        :type shuffle: bool
        :param optimizerCreator: Optimizer creator that will create optimizer that should be used for training.
            Send None when training is not your case.
        :type optimizerCreator: Optional[OptimizerCreator]
        :param loadingWorkers: Number of parallel workers that will be used for dataset loading.
            Value greater than one activates parallel processing.
        :type loadingWorkers: int
        :param forceDevice: Name of device that should be forced to torch. Default is none which means that it uses cuda
            if can and cpu otherwise.
        :type forceDevice: str
        :param dataLoaderCollate: You can force own collate method for data loader.
        :type dataLoaderCollate: Optional[Callable]
        :param modelTrain: Flag that determines mode of model. True -> Train. False -> eval
        :type modelTrain: bool
        :param sampler: Sampler that should by used by the data loader. If you use these parameter do not forget to
            set the shuffle to False, because torch requires it.
        :type sampler: Sampler
        :return: Returns data loader, optimizer (in case of provided creator) and device where the
            model is.
        :rtype: Tuple[DataLoader, torch.optim.Optimizer, torch.device]
        :raise ImportError: When Nvidia apex couldn't be imported and therefore fp16 precision can not be used.
                    Screams only when you ask for fp16 precision.
        :raise AttributeError: This model is in mixed precision and can be used only with cuda device.
        """

        dataLoader = DataLoader(dataset, batch_size=batchSize,
                                shuffle=shuffle, num_workers=loadingWorkers, collate_fn=dataLoaderCollate,
                                sampler=sampler)

        optimizer, device = self._initModel(optimizerCreator, forceDevice, modelTrain)

        return dataLoader, optimizer, device

    def _initModel(self, optimizerCreator: Optional[OptimizerCreator] = None, forceDevice: str = None,
                            modelTrain: bool = False) -> Tuple[Optional[torch.optim.Optimizer], torch.device]:
        """
        Initialization method that gathers common operations that are done at start of train, eval or inference phase.

        :param optimizerCreator: Optimizer creator that will create optimizer that should be used for training.
            Send None when training is not your case.
        :type optimizerCreator: Optional[OptimizerCreator]
        :param forceDevice: Name of device that should be forced to torch. Default is none which means that it uses cuda
            if can and cpu otherwise.
        :type forceDevice: str
        :param modelTrain: Flag that determines mode of model. True -> Train. False -> eval
        :type modelTrain: bool
        :return: Returns  optimizer (in case of provided creator) and device where the
            model is.
        :rtype: Tuple[Optional[torch.optim.Optimizer], torch.device]
        :raise ImportError: When Nvidia apex couldn't be imported and therefore fp16 precision can not be used.
                    Screams only when you ask for fp16 precision.
        :raise AttributeError: This model is in mixed precision and can be used only with cuda device.
        """

        if forceDevice is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(forceDevice)

        self.model = self.model.to(device)

        # Let's create user selected optimizer
        optimizer = None if optimizerCreator is None else optimizerCreator.create(self.model)

        if device.type != "cuda" and (self._shouldActivateMixedPrecision or self._mixedPrecisionActivated):
            raise AttributeError("This model is in mixed precision and can be used only with cuda device.")

        if device.type == "cuda" and self._shouldActivateMixedPrecision:
            try:
                from apex import amp
                amp.register_float_function(torch, 'sigmoid')
                if not self._mixedPrecisionActivated:
                    self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.FP16_MIX_PREC_OPT_LVL.value)
                    self._mixedPrecisionActivated = True
                    self._shouldActivateMixedPrecision = False
            except ImportError:
                raise ImportError("You must install apex (https://www.github.com/nvidia/apex) for fp16 training.")

        if modelTrain:
            self.model.train()
        else:
            self.model.eval()

        return optimizer, device

    def train(self, dataset: Dataset, epochs: int, batchSize: int, accuGrad: int = 1, shuffle: bool = True,
              optimizerCreator: OptimizerCreator = BERTAdamWOptimizerCreator(), loadingWorkers: int = 1,
              forceDevice: str = None, verbose: bool = True, endOfEpoch: Optional[CoachTransformersValidateCallback] = None,
              subsampling: float = -1, useClassWeights: bool= False, accuGradNorm: bool = True):
        """
        Trains on given dataset.

        :param dataset: Dataset that should be used for training.
            If that dataset implements own collate_fn method than it will be automatically used.
        :type dataset: Dataset
        :param epochs: Number of training epochs.
        :type epochs: int
        :param batchSize: Size of one batch.
        :type batchSize: int
        :param accuGrad: Values greater than one activates gradient accumulation. Gradient from batches will be
            accumulated for X steps and after that we will perform the optimization, which means, from the optimization
            point of view, that we have virtually batches of size X*BATCH_SIZE.
            Also see accuGradNorm.
        :type accuGrad: int
        :param shuffle: True reshuffles data on every epoch.
        :type shuffle: bool
        :param optimizerCreator: Optimizer creator that will create optimizer that should be used for training.
        :type optimizerCreator: OptimizerCreator
        :param loadingWorkers: Number of parallel workers that will be used for dataset loading.
            Value greater than one activates parallel processing.
        :type loadingWorkers: int
        :param forceDevice: Name of device that should be forced to torch. Default is none which means that it uses cuda
            if can and cpu otherwise.
        :type forceDevice: str
        :param verbose: True activates printing of information about learning.
        :type verbose: bool
        :param endOfEpoch: Callback that should be called after the end of each epoch.
            First argument, of the callback, receives epoch number (starts from zero).
            The callback should return boolean that determines if the training should be early stopped (true means stop).
        :type endOfEpoch: Optional[CoachTransformersValidateCallback]
        :param subsampling: On the beginning of each epoch subset of train set is selected. Resampling without replacement is performed (a sample can not be selected two times).
            Numbers <= 0 deactivates this feature (default). If x (x is actual argument) is in range (0,1) than we select x*TRAIN_SAMPLES samples. If x >= 1 than we select
            x samples (choose the x in [1, TRAIN_SAMPLES]).
        :type subsampling: float
        :param useClassWeights: True uses class weights (usefull for imbalanced dataset) if the dataset provides them and the model
            can use them. Practicaly it means that the dataset must have the classWeights property and the model must have it also.
        :type useClassWeights: bool
        :param accuGradNorm: Activates gradient normalization by number of accuGrads. Use eg. when you are using the mean for your loss.
        :type accuGradNorm: bool
        """
        if verbose:
            Logger().log("Starts training")

        if useClassWeights:
            # we want to use class weighting
            Logger().log("I am setting the class weights: "+", ".join(str(w) for w in dataset.classWeights))
            self.model.classWeights = dataset.classWeights

        sampler = None
        if subsampling > 0:
            shuffle = False  # torch requires that if you wan to use own sampler

            sampler = IndicesSubsampler(source=dataset,
                                 subsetLen=int(len(dataset) * subsampling) if subsampling < 1 else int(subsampling))

        dataLoader, optimizer, device = self._initModelAndLoader(dataset=dataset,
                                                                        batchSize=batchSize,
                                                                        shuffle=shuffle,
                                                                        optimizerCreator=optimizerCreator,
                                                                        loadingWorkers=loadingWorkers,
                                                                        forceDevice=forceDevice,
                                                                        modelTrain=True,
                                                                        dataLoaderCollate=dataset.collate_fn if callable(getattr(dataset, "collate_fn", None)) else None,
                                                                        sampler=sampler)

        numberOfIterations = ceil((len(dataLoader) / accuGrad))

        if self._mixedPrecisionActivated:
            if verbose:
                Logger().log("FP16 activated")

            from apex import amp
            # there is no need to wrap the import into try - except like in _initModelAndLoader, because the
            # _initModelAndLoader already tries to import that.

        iBatch = 0

        actAccuGrad = accuGrad
        optimizer.zero_grad()  # just to be sure we are clean

        for epoch in range(epochs):

            iterCnt = 0

            epochLossAccumulator = 0.0
            iterLossAccumulator = 0.0

            with tqdm(total=numberOfIterations, desc="epoch {}".format(epoch + 1), leave=True,
                      unit="iterations") as pBar:

                pBarTime = time.time()
                updateBar = 0
                for iBatch, samplesInBatch in enumerate(dataLoader):

                    dataMap = {attr: samplesInBatch[valueOffset].to(device) for attr, valueOffset in self.trainInput2DataMap.items()}

                    # let's send the tensors to device and do the forward pass
                    loss = self.model(**dataMap)[0]

                    if accuGradNorm:
                        # normalize the loss
                        if iBatch + actAccuGrad > len(dataLoader):
                            # we are not able to have whole virtual batch
                            normBy = len(dataLoader) - (iBatch - (accuGrad - actAccuGrad))
                        else:
                            # this batch will be whole
                            normBy = accuGrad
                        loss = loss / normBy

                    # calculate gradients according to calculated loss
                    if device.type == "cuda" and self._mixedPrecisionActivated:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    actAccuGrad -= 1
                    iterLossAccumulator += loss.item()
                    if actAccuGrad == 0 or iBatch == len(dataLoader) - 1:
                        epochLossAccumulator += iterLossAccumulator

                        iterCnt += 1
                        updateBar += 1
                        if time.time() - pBarTime > self.BAR_INTERVAL:
                            pBar.set_description("epoch {} | loss in last iteration: {}".format(
                                epoch + 1, iterLossAccumulator
                            ))
                            pBar.update(updateBar)
                            updateBar = 0
                            pBarTime = time.time()

                        # update weights
                        optimizer.step()

                        # Zero the accumulated parameters gradient, because we are going to calc a new one.
                        optimizer.zero_grad()

                        actAccuGrad = accuGrad
                        iterLossAccumulator = 0

            averageLoss = epochLossAccumulator / iterCnt
            if verbose:
                Logger().log("Finished epoch {}. | average loss {}".format(epoch + 1, averageLoss))

            if endOfEpoch is not None:
                if endOfEpoch(epoch, averageLoss, save=True):
                    # callback says stop
                    if verbose:
                        Logger().log("Early stopping the training after {}. epoch.".format(epoch + 1))
                    break

                # the callback may change the model configuration so let's init it again
                _, device = self._initModel(forceDevice=forceDevice, modelTrain=True)

        if verbose:
            Logger().log("Finishes training")

    def useModel(self, dataset: Dataset, batchSize: int, loadingWorkers: int = 1,
                forceDevice: str = None, progressBarDesc: str = "Working", progressBarUnit: str = "sample",
                transferResToCpu: bool = True) -> Generator[Any, None, None]:
        """
        Use trained model on given dataset samples. For each title edit measures how funny it is.

        :param dataset: Dataset you want to use.
            If that dataset implements own collate_fn method than it will be automatically used.
        :type dataset: Dataset
        :param batchSize: Size of one batch.
        :type batchSize: int
        :param loadingWorkers: Number of parallel workers that will be used for dataset loading.
          Value greater than one activates parallel processing.
        :type loadingWorkers: int
        :param forceDevice: Name of device that should be forced to torch. Default is none which means that it uses cuda
          if can and cpu otherwise.
        :type forceDevice: str
        :param progressBarDesc: Description that will be used for progress bar.
        :type progressBarDesc: str
        :param progressBarUnit: Name of unit that will show the progress bar.
        :type progressBarUnit: str
        :param transferResToCpu: If True than the results from the model will be transferred to cpu.
            If False the device of results remains the same as model device.
        :type transferResToCpu: bool
        :return: Generator that generates results of the model per sample in original order.
        :rtype: Generator[Any, None, None]
        """

        dataLoader, _, device = self._initModelAndLoader(dataset=dataset,
                                                         batchSize=batchSize,
                                                         shuffle=False,
                                                         optimizerCreator=None,
                                                         loadingWorkers=loadingWorkers,
                                                         forceDevice=forceDevice,
                                                         modelTrain=False,
                                                         dataLoaderCollate=dataset.collate_fn if callable(getattr(dataset, "collate_fn", None)) else None,)

        with torch.no_grad():
            updateBar = 0
            pBarTime = time.time()

            with tqdm(total=len(dataset), desc=progressBarDesc, leave=True,
                      unit=progressBarUnit) as pBar:

                for samplesInBatch in dataLoader:
                    dataMap = {attr: samplesInBatch[valueOffset].to(device) for attr, valueOffset in
                               self.useInput2DataMap.items()}
                    # forward pass
                    res = self.model(**dataMap)

                    if transferResToCpu:
                        if isinstance(res, tuple):
                            resNew = []
                            for r in res:
                                resNew.append(r.cpu())
                            res = resNew
                        else:
                            res = res.cpu()

                    for x in zip(*res) if isinstance(res, tuple) else res:
                        updateBar += 1
                        if time.time() - pBarTime > self.BAR_INTERVAL:
                            pBar.update(updateBar)
                            updateBar = 0
                            pBarTime = time.time()

                        yield x
