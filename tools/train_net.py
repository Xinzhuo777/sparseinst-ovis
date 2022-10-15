import itertools
from typing import Any, Dict, List, Set
from detectron2 import data
import torch
from collections import OrderedDict
import logging
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, build_detection_test_loader, DatasetMapper,DatasetCatalog
from detectron2.engine import AutogradProfiler, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
)

import os,sys
sys.path.append(os.path.dirname(os.path.realpath('/home/featurize/sparseinstovis/data_video')))

from data_video.ytvis_eval import YTVISEvaluator
from data_video.dataset_mapper import YTVISDatasetMapper
from data_video.build import get_detection_dataset_dicts,build_detection_train_loader

sys.path.append(".")
from sparseinst import add_sparse_inst_config, COCOMaskEvaluator
sys.path.append(os.path.dirname(os.path.realpath('/home/featurize/sparseinstovis/data_video/datasets')))
from datasets.ytvis import (
    register_ytvis_instances,
    _get_ovis_instances_meta,
)



class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            os.makedirs(output_folder, exist_ok=True)

        return YTVISEvaluator(dataset_name, cfg, True, output_folder)
    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #     """
    #     Create evaluator(s) for a given dataset.
    #     This uses the special metadata "evaluator_type" associated with each builtin dataset.
    #     For your own dataset, you can simply create an evaluator manually in your
    #     script and do not have to worry about the hacky if-else logic here.
    #     """
    #     if output_folder is None:
    #         output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    #     evaluator_list = []
    #     evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    #     if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
    #         evaluator_list.append(
    #             SemSegEvaluator(
    #                 dataset_name,
    #                 distributed=True,
    #                 num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
    #                 ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
    #                 output_dir=output_folder,
    #             )
    #         )
    #     if evaluator_type in ["coco", "coco_panoptic_seg"]:
    #         evaluator_list.append(COCOMaskEvaluator(dataset_name, ("segm", ), True, output_folder))
    #     if evaluator_type == "coco_panoptic_seg":
    #         evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    #     if evaluator_type == "cityscapes_instance":
    #         assert (
    #             torch.cuda.device_count() >= comm.get_rank()
    #         ), "CityscapesEvaluator currently do not work with multiple machines."
    #         return CityscapesInstanceEvaluator(dataset_name)
    #     if evaluator_type == "cityscapes_sem_seg":
    #         assert (
    #             torch.cuda.device_count() >= comm.get_rank()
    #         ), "CityscapesEvaluator currently do not work with multiple machines."
    #         return CityscapesSemSegEvaluator(dataset_name)
    #     elif evaluator_type == "pascal_voc":
    #         return PascalVOCDetectionEvaluator(dataset_name)
    #     elif evaluator_type == "lvis":
    #         return LVISEvaluator(dataset_name, cfg, True, output_folder)
    #     if len(evaluator_list) == 0:
    #         raise NotImplementedError(
    #             "no Evaluator for the dataset {} with the type {}".format(
    #                 dataset_name, evaluator_type
    #             )
    #         )
    #     elif len(evaluator_list) == 1:
    #         return evaluator_list[0]
    #     return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            # for transformer
            if "patch_embed" in key or "cls_token" in key:
                weight_decay = 0.0
            if "norm" in key:
                weight_decay = 0.0
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full  model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR, amsgrad=cfg.SOLVER.AMSGRAD
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_test_loader(cls, cfg,dataset_name):
        dataset_name = cfg.DATASETS.TEST[0]
        if cfg.MODEL.SPARSE_INST.DATASET_MAPPER == "SparseInstDatasetMapper":
            from sparseinst import SparseInstDatasetMapper
            mapper = SparseInstDatasetMapper(cfg, is_train=True)
        else:
            mapper = None
        return build_detection_test_loader(cfg,dataset_name, mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_name = cfg.DATASETS.TRAIN[0]
        if cfg.MODEL.SPARSE_INST.DATASET_MAPPER == "SparseInstDatasetMapper":
            from sparseinst import SparseInstDatasetMapper
            mapper = SparseInstDatasetMapper(cfg, is_train=True)
        else:
            mapper = None
        dataset_dict = get_detection_dataset_dicts(
            dataset_name,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        return build_detection_train_loader(cfg, mapper=mapper)     

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
        Returns:
            dict: a dict of result metrics
        """
        from torch.cuda.amp import autocast
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg,dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            with autocast():
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    print(args.config_file)
    add_sparse_inst_config(cfg)
    print(args.config_file)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "sparseinst" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="sparseinst")
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("OVIS/train",
                         "OVIS/annotations/annotations_train.json"),
    "ovis_val": ("OVIS/valid",
                       "OVIS/annotations/annotations_valid.json"),
    "ovis_test": ("OVIS/test",
                        "OVIS/annotations/annotations_test.json"),
}

def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        print(json_file)
        print(image_root)
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

if __name__ == "__main__":
    # dataset_name0 = 'ovis_train'
    # dataset_name1 = 'ovis_val'
    # dataset_name2 = 'ovis_test'
    # MetadataCatalog.remove(dataset_name0)
    # DatasetCatalog.remove(dataset_name0)
    # MetadataCatalog.remove(dataset_name1)
    # DatasetCatalog.remove(dataset_name1)
    # MetadataCatalog.remove(dataset_name2)
    # DatasetCatalog.remove(dataset_name2)

    _root = "/home/featurize"
    register_all_ovis(_root)

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
