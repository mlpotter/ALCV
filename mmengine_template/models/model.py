"""This module is used to implement and register the custom model.

Follow the `guide <https://mmengine.readthedocs.io/en/latest/tutorials/model.html>`
in MMEngine to implement CustomModel

The default implementation only does the register process. Users need to rename
the ``CustomModel`` to the real name of the model and implement it.
"""  # noqa: E501
from mmengine.model import BaseModel

from mmdet import models
from mmengine_template.registry import MODELS
from mmengine_template.registry import DETECTORS
# from mmdet.registry import MODELS as MMDET_MODELS
# from mmdet.registry import TASK_UTILS


from collections.abc import Sequence
import torch
from mmdet.structures.bbox import bbox2roi


@MODELS.register_module()
class MixedModel(BaseModel):
    def __init__(self, model: dict, data_preprocessor=dict,train_cfg=None, test_cfg=None):
        super(MixedModel, self).__init__()

        model.data_preprocessor = data_preprocessor
        self.detector = DETECTORS.build(model)
        self.data_preprocessor = MODELS.build(data_preprocessor)

        if train_cfg is not None:
            pass
        elif test_cfg is not None:  # only applies during test
            pass


    def forward(self,
                inputs,
                data_samples=None,
                mode='tensor',**kwargs):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
            - If ``mode="active"``, return a dict of AL scores with meta information
        """
        if mode == 'loss':
            loss = self.forward_train(inputs,data_samples,**kwargs)
            return loss #self.loss(inputs, data_samples)

        elif mode == 'predict':
            return self.detector.predict(inputs, data_samples)

        elif mode == 'tensor':
            return self.detector._forward(inputs, data_samples)

        elif mode == 'active':
            return self.forward_active_learning(inputs,data_samples)

        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def forward_active_learning(self,inputs,data_samples,**kwargs):
        results = {}
        # pred_cls is Number of proposals after NMS x (80+1)
        # pred_reg is Number of proposals after NMS x (80 * 4 , x y dx dy)
        # maybe NMS only applies during training , not inference?
        # tuple of feature pyramid levels ( Tensor, Tensor, Tensor , ... Tensor) tuple size is number of FPN levels
        # Tensor is Bx256xFPWxFPH
        feat_fpn = self.detector.extract_feat(inputs)

        # tuple of ( InstanceData[instance] , ... , InstanceData[instance] ) tuple size is number of input images
        # instance [bbox, label, objectness score]
        # bbox [x,y,x2,y2]? rescale=True return boxes in original image space
        instances_rpn = self.detector.rpn_head.predict(feat_fpn, data_samples, rescale=False)

        proposals = [rpn_results.bboxes for rpn_results in instances_rpn]
        rois = bbox2roi(proposals)
        # bbox head
        if self.detector.roi_head.with_bbox:
            bbox_results = self.detector.roi_head._bbox_forward(feat_fpn, rois)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)


        # (B*NMS/MaxBBox x num class + background) , (B x num_classes*4) bounding box offset, (B x NMS/MaxBBox x 7 x 7) roi align pool?
        # cls_prob, bbox_offset, bbox_roi_feat = bbox_results.values()

        # tuple of (tensor [B*NMS/BoxMax x NumClass, tensor [B*NMS/BoxMax x NumClass*4)] )
        # roi_outs = self.detector.roi_head.predict(feat_fpn,instances_rpn,data_samples,rescale=False)
        # last dim of classes may be score (background vs non-background)
        # self.detector.predict(inputs,data_samples)

        results = dict(img_id=[],score=[],meta=[])
        for i,(image_cls_scores,meta_data) in enumerate(zip(cls_score,data_samples)):

            image_cls_probabilities = torch.softmax(image_cls_scores,axis=-1)
            image_obj_entropies = -torch.sum(image_cls_probabilities * torch.log(image_cls_probabilities),axis=-1)
            image_entropy = torch.mean(image_obj_entropies)

            results['img_id'].append(meta_data.img_id)
            results['score'].append(image_entropy.item())
            results['meta'].append({k: meta_data.get(k,None) for k in ['img_path','img_shape','scale_factor','ori_shape','img_id','batch_input_shape']})

        return results


    def forward_train(self, inputs, data_samples, **kwargs):
        loss = {}

        if 'sup' in inputs.keys(): #inputs.get('sup',False):
            loss_sup = self.detector(inputs['sup'],data_samples['sup'],mode='loss')
            loss.update({f"{k}_sup":v for k,v in loss_sup.items()})

        if 'unsup_weak' in inputs.keys(): #get('unsup_weak',False):
            # pred_cls is Number of proposals after NMS x (80+1)
            # pred_reg is Number of proposals after NMS x (80 * 4 , x y dx dy)
            pred_cls,pred_reg = self.detector.forward(inputs['unsup_weak'],data_samples['unsup_weak'],mode='tensor')[0]

        #
        # kwargs.update({"img": img})
        # kwargs.update({"img_metas": img_metas})
        # kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        # # create a dictionary with keys ['partial_strong','partial_weak','labeled'] such that images are divided by this tag
        # # this is done using dict_split
        # data_groups = dict_split(kwargs, "tag")
        # for _, v in data_groups.items():
        #     v.pop("tag")
        #
        # loss = {}

        return loss

