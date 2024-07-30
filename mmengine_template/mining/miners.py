from mmengine.evaluator import BaseMetric
from .registry import MINERS

import ujson as json
import os
import os.path as osp
import numpy as np
import logging
from tqdm import tqdm
import torch
import random
from mmdet.structures.bbox import bbox2roi
from mmengine.utils import ProgressBar
@MINERS.register_module()
class CustomAL(BaseMetric):

    def add(self, gt, preds):
        ...

    # NOTE for evaluator
    def compute_metric(self, size):
        ...

def al_scores_single_gpu(model,
                    data_loader,
                    logger,
                    save_dir,
                    active_cycle=-1,
                    **kwargs):
    model.eval()
    results = dict(img_id=[], score=[], meta=[])
    logger.info("Acquiring Active Learning Scores")
    progress_bar = ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data = model.data_preprocessor(data, False) # MLP: stop point

            result = model.forward(**data,
                                   mode='active')

            progress_bar.update()

        results = {key: results[key] + result[key] for key in results.keys()}
    return results


@MINERS.register_module()
class JsonWriter(object):
    def __init__(self,data_file,initial_labeled_size=1000,labeled_size=1000):
        self.labeled_size = labeled_size
        self.initial_labeled_size = initial_labeled_size
        self.logger = None

        self.unlabeled_set = set()
        self.labeled_set = set()

        self.data_file = data_file

    def set_logger(self,logger):
        self.logger = logger

    def create_initial_data_partition(self,save_dir):

        f = open(self.data_file)
        self.dataset_json = json.load(f)
        f.close()

        images, annotations, info, categories, license =  self.json_separate(self.dataset_json)

        self.total_images = set([img['id'] for img in images]) #i for i in range(N_images)])
        self.labeled_set.update(random.sample(self.total_images,self.initial_labeled_size))
        self.unlabeled_set.update(self.total_images.difference(self.labeled_set))

        labeled_json_file = osp.join(save_dir,"labeled.json")
        unlabeled_json_file = osp.join(save_dir,"unlabeled.json")

        labeled_json, unlabeled_json = self.create_labeled_unlabeled( images, annotations, info, categories, license)

        with open(labeled_json_file,'w') as f:
            json.dump(labeled_json,f)

        with open(unlabeled_json_file,'w') as f:
            json.dump(unlabeled_json,f)

        if self.logger is not None:
            self.logger.info(f"Saved labeled and unlabeled json file to {save_dir} for AL cycle {self.al_cycle}")
            self.logger.info(f"Total Images in dataset: {len(self.total_images)}")
            self.logger.info("Labeled / Unlabeled Dataset Size: {} / {}".format(len(self.labeled_set),len(self.unlabeled_set)))

    def create_data_partitions(self,scores_file,save_dir):
        self.logger.info(f"Load AL scores from {scores_file}")

        f = open(scores_file)
        scores_json = json.load(f)
        f.close()

        scores = np.array(scores_json['score'])
        img_ids = np.array(scores_json['img_id'])
        metas = scores_json['meta']

        top_scores_idx = np.argpartition(-scores, self.labeled_size - 1)

        top_img_ids = img_ids[top_scores_idx].tolist()[:self.labeled_size]

        self.labeled_set.update(top_img_ids)
        self.unlabeled_set = self.unlabeled_set.difference(top_img_ids)

        images, annotations, info, categories, license =  self.json_separate(self.dataset_json)
        labeled_json, unlabeled_json = self.create_labeled_unlabeled( images, annotations, info, categories, license)

        labeled_json_file = osp.join(save_dir, "labeled.json")
        unlabeled_json_file = osp.join(save_dir, "unlabeled.json")

        with open(labeled_json_file, 'w') as f:
            json.dump(labeled_json, f)

        with open(unlabeled_json_file, 'w') as f:
            json.dump(unlabeled_json, f)

        if self.logger is not None:
            self.logger.info(f"Saved labeled and unlabeled json file to {save_dir} for AL cycle {self.al_cycle}")
            self.logger.info(f"Total Images in dataset: {len(self.total_images)}")
            self.logger.info("Labeled / Unlabeled Dataset Size: {} / {}".format(len(self.labeled_set),len(self.unlabeled_set)))


    def create_labeled_unlabeled(self,images,annotations,info,categories,license):
        labeled_json = dict.fromkeys(["images","annotations","info","categories","license"])
        labeled_json['images'] = [img for img in images if img['id'] in self.labeled_set] #images[idx] for idx in labeled_idx]
        labeled_json['annotations'] = [annot for annot in annotations if annot['image_id'] in self.labeled_set]
        labeled_json['info'] = info
        labeled_json['categories'] = categories
        labeled_json['license'] = license

        unlabeled_json = dict.fromkeys(["images","annotations","info","categories","license"])
        unlabeled_json['images'] = [img for img in images if img['id'] in self.unlabeled_set]
        unlabeled_json['annotations'] = [annot for annot in annotations if annot['image_id'] in self.unlabeled_set]
        unlabeled_json['info'] = info
        unlabeled_json['categories'] = categories
        unlabeled_json['license'] = license

        return labeled_json,unlabeled_json

    def json_separate(self,dataset_json):
        images= dataset_json['images']
        annotations = dataset_json['annotations']
        info = dataset_json['info']
        categories = dataset_json['categories']
        license = dataset_json['license']

        return images,annotations,info,categories,license


@MINERS.register_module()
class JsonWriterOld(object):
    def __init__(self, data_file, initial_labeled_size=1000, labeled_size=1000, al_cycle=0):
        self.labeled_size = labeled_size
        self.initial_labeled_size = initial_labeled_size
        self.al_cycle = al_cycle
        self.logger = None

        self.unlabeled_set = set()
        self.labeled_set = set()

        self.data_file = data_file

    def set_logger(self, logger):
        self.logger = logger

    def create_initial_data_partition(self, save_dir):

        f = open(self.data_file)
        dataset_json = json.load(f)
        f.close()

        images, annotations, info, categories, license = self.json_separate(dataset_json)

        self.total_images = set([img['id'] for img in images])  # i for i in range(N_images)])

        self.labeled_set.update(random.sample(self.total_images, self.initial_labeled_size))
        self.unlabeled_set.update(self.total_images.difference(self.labeled_set))

        labeled_json_file = osp.join(save_dir, "labeled.json")
        unlabeled_json_file = osp.join(save_dir, "unlabeled.json")

        labeled_json, unlabeled_json = self.create_labeled_unlabeled(images, annotations, info, categories, license)

        with open(labeled_json_file, 'w') as f:
            json.dump(labeled_json, f)

        with open(unlabeled_json_file, 'w') as f:
            json.dump(unlabeled_json, f)

        if self.logger is not None:
            self.logger.info(f"Saved labeled.json and unlabeled.json file to {save_dir} for AL cycle {self.al_cycle}")
            self.logger.info(f"Total Images in dataset: {len(self.total_images)}")
            self.logger.info(
                "Labeled / Unlabeled Dataset Size: {} / {}".format(len(self.labeled_set), len(self.unlabeled_set)))

    def create_data_partitions(self, base_work_dir, scores_file, save_dir):
        self.logger.info(f"Load AL scores from {scores_file}")

        f = open(scores_file)
        scores_json = json.load(f)
        f.close()

        f = open(osp.join(base_work_dir, f"step_{self.al_cycle - 1}", "labeled.json"))
        prev_labeled_json = json.load(f)
        f.close()

        f = open(osp.join(base_work_dir, f"step_{self.al_cycle - 1}", "unlabeled.json"))
        prev_unlabeled_json = json.load(f)
        f.close()

        f = open(self.data_file)
        dataset_json = json.load(f)
        f.close()

        self.labeled_set = set([img.get('id') for img in prev_labeled_json['images']])
        self.unlabeled_set = set([img.get('id') for img in prev_unlabeled_json['images']])
        self.total_images = self.labeled_set.union(self.unlabeled_set)

        scores = np.array(scores_json['score'])
        img_ids = np.array(scores_json['img_id'])
        metas = scores_json['meta']

        top_scores_idx = np.argpartition(-scores, self.labeled_size - 1)

        top_img_ids = img_ids[top_scores_idx].tolist()

        self.labeled_set.update(top_img_ids)
        self.unlabeled_set = self.unlabeled_set -set(top_img_ids)

        images, annotations, info, categories, license = self.json_separate(dataset_json)
        labeled_json, unlabeled_json = self.create_labeled_unlabeled(images, annotations, info, categories, license)

        labeled_json_file = osp.join(save_dir, "labeled.json")
        unlabeled_json_file = osp.join(save_dir, "unlabeled.json")

        with open(labeled_json_file, 'w') as f:
            json.dump(labeled_json, f)

        with open(unlabeled_json_file, 'w') as f:
            json.dump(unlabeled_json, f)

        if self.logger is not None:
            self.logger.info(f"Saved labeled.json and unlabeled.json file to {save_dir} for AL cycle {self.al_cycle}")
            self.logger.info(f"Total Images in dataset: {len(self.total_images)}")
            self.logger.info(
                "Labeled / Unlabeled Dataset Size: {} / {}".format(len(self.labeled_set), len(self.unlabeled_set)))

    def create_labeled_unlabeled(self, images, annotations, info, categories, license):
        labeled_json = dict.fromkeys(["images", "annotations", "info", "categories", "license"])
        labeled_json['images'] = [img for img in images if
                                  img['id'] in self.labeled_set]  # images[idx] for idx in labeled_idx]
        labeled_json['annotations'] = [annot for annot in annotations if annot['image_id'] in self.labeled_set]
        labeled_json['info'] = info
        labeled_json['categories'] = categories
        labeled_json['license'] = license

        unlabeled_json = dict.fromkeys(["images", "annotations", "info", "categories", "license"])
        unlabeled_json['images'] = [img for img in images if img['id'] in self.unlabeled_set]
        unlabeled_json['annotations'] = [annot for annot in annotations if annot['image_id'] in self.unlabeled_set]
        unlabeled_json['info'] = info
        unlabeled_json['categories'] = categories
        unlabeled_json['license'] = license

        return labeled_json, unlabeled_json

    def json_separate(self, dataset_json):
        images = dataset_json['images']
        annotations = dataset_json['annotations']
        info = dataset_json['info']
        categories = dataset_json['categories']
        license = dataset_json['license']

        return images, annotations, info, categories, license
# return labeled_json,unlabeled_json