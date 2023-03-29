import pytest
from icevision.all import *

class BinaryJaccardIndex(Metric):
    """Jaccard Index for Semantic Segmentation
    Calculates Jaccard Index for semantic segmentation (binary images only).
    """

    def __init__(self):
        self._reset()

    def _reset(self):
        self._union = 0
        self._intersection = 0

    def accumulate(self, preds):

        pred = (
            np.stack([x.pred.segmentation.mask_array.data for x in preds])
            .astype(np.bool)
            .flatten()
        )

        target = (
            np.stack([x.ground_truth.segmentation.mask_array.data for x in preds])
            .astype(np.bool)
            .flatten()
        )

        self._union += np.logical_or(pred, target).sum()
        self._intersection += np.logical_and(pred, target).sum()

    def finalize(self) -> Dict[str, float]:

        if self._union == 0:
            jaccard = 0

        else:
            jaccard = self._intersection / self._union

        self._reset()
        return {"binary_jaccard_value_for_fastai": jaccard}


@pytest.fixture()
def setup_prediction():
    # synthetic data to test
    gt = np.asarray([
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
    ])

    pred = np.asarray([
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,1,0,0,0,0,0],
        [0,0,0,0,0,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0],
    ])

    # setup pred record
    pred_record = BaseRecord((
        ImageRecordComponent(),
        SemanticMaskRecordComponent(),
        ClassMapRecordComponent(task=tasks.segmentation),))

    pred_record.segmentation.set_class_map(ClassMap(["square"]))
    pred_record.segmentation.set_mask_array(MaskArray(pred))

    # setup ground truth record
    gt_record = BaseRecord((
        ImageRecordComponent(),
        SemanticMaskRecordComponent(),
        ClassMapRecordComponent(task=tasks.segmentation),)) 

    gt_record.segmentation.set_class_map(ClassMap(["square"]))
    gt_record.segmentation.set_mask_array(MaskArray(gt))

    # w, h = imgA.shape[0], imgA.shape[1]
    w, h = gt.shape[0], gt.shape[1]

    gt_record.set_img_size(ImgSize(w,h), original=True)

    prediction = Prediction(pred=pred_record, ground_truth=gt_record)

    return prediction


# testing metric 
@pytest.fixture()
def expected_binary_jaccard_output():

    return {'binary_jaccard_value_for_fastai': 0.25}


# @pytest.fixture()
def test_binary_jaccard_metric(setup_prediction, expected_binary_jaccard_output):

    pred = setup_prediction

    jaccard = BinaryJaccardIndex()
    jaccard.accumulate([pred])

    assert jaccard.finalize() == expected_binary_jaccard_output
